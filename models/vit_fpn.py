
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTMultiScaleExtractor(nn.Module):
    EXTRACT_INDICES = [2, 5, 8, 11]

    def __init__(
        self,
        model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=True,
        img_size=640,
        freeze_backbone_epochs=5,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=img_size,
        )
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.embed_dim = self.backbone.embed_dim
        if freeze_backbone_epochs > 0:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        B, C, H, W = x.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        x_tokens = self.backbone.patch_embed(x)
        x_tokens = self.backbone._pos_embed(x_tokens)
        x_tokens = self.backbone.patch_drop(x_tokens)
        x_tokens = self.backbone.norm_pre(x_tokens)
        features = []
        for i, block in enumerate(self.backbone.blocks):
            x_tokens = block(x_tokens)
            if i in self.EXTRACT_INDICES:
                patch_tokens = x_tokens[:, 1:, :]
                spatial = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, ph, pw)
                features.append(spatial)
        return features


class FPN(nn.Module):
    def __init__(self, in_channels=768, out_channels=256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(4)])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(4)])

    def forward(self, features):
        lat = [l(f) for l, f in zip(self.lat, features)]
        out = [lat[3]]
        for i in range(2, -1, -1):
            upsampled = F.interpolate(out[-1], size=lat[i].shape[-2:], mode="nearest")
            out.append(self.smooth[i](lat[i] + upsampled))
        return list(reversed(out))


class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1),
        )

    def forward(self, x):
        return self.cls_head(x), self.reg_head(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, p2, target_size):
        return F.interpolate(self.decoder(p2), size=target_size, mode="bilinear", align_corners=False)


class DigitalStewardViTFPN(nn.Module):
    def __init__(self, num_classes=2, fpn_out_channels=256, freeze_backbone_epochs=5, pretrained=True, img_size=640):
        super().__init__()
        self.extractor = ViTMultiScaleExtractor(
            pretrained=pretrained,
            freeze_backbone_epochs=freeze_backbone_epochs,
            img_size=img_size,
        )
        self.fpn = FPN(in_channels=self.extractor.embed_dim, out_channels=fpn_out_channels)
        self.det_heads = nn.ModuleList([DetectionHead(fpn_out_channels, num_classes) for _ in range(4)])
        self.seg_head = SegmentationHead(fpn_out_channels, num_classes=2)

    def forward(self, x):
        H, W = x.shape[-2:]
        vit_features = self.extractor(x)
        pyramid = self.fpn(vit_features)
        cls_maps, reg_maps = [], []
        for head, feat in zip(self.det_heads, pyramid):
            cls, reg = head(feat)
            cls_maps.append(cls)
            reg_maps.append(reg)
        seg_mask = self.seg_head(pyramid[0], (H, W))
        return {"cls_maps": cls_maps, "reg_maps": reg_maps, "seg_mask": seg_mask}

    def unfreeze_backbone(self):
        self.extractor.unfreeze_backbone()

    @torch.no_grad()
    def predict(self, x, conf_thresh=0.45):
        out = self.forward(x)
        return {
            "cls_confidence": [m.softmax(dim=1) for m in out["cls_maps"]],
            "reg_maps": out["reg_maps"],
            "seg_mask": out["seg_mask"].argmax(dim=1),
        }
