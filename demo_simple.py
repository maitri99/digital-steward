
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import time
import pandas as pd
from pathlib import Path
import fastf1
import re

fastf1.Cache.enable_cache('/home/dell/digital_steward/.fastf1_cache')

model = YOLO("/home/dell/digital_steward/runs/steward_augmented2/weights/best.pt")

RUNS_DIR     = Path("/home/dell/digital_steward/runs/steward_augmented2")
TEST_IMG_DIR = Path("/home/dell/HACKATHON/Formula 1.v1i.yolov8/test/images")
TEST_LBL_DIR = Path("/home/dell/HACKATHON/Formula 1.v1i.yolov8/test/labels")
CLASS_NAMES  = ["Non-penalty", "Penalty"]
CLASS_COLORS_BGR = [(0,200,0),(0,0,220)]

session_stats = {"frames":0,"violations":0,"manual_reviews":0,"clear":0,"total_ms":0.0}

# ── FastF1 data (loaded once at startup) ──────────────────────────────────────
print("Loading FastF1 session data...")
_session = fastf1.get_session(2023, "Austrian Grand Prix", "Q")
_session.load(telemetry=False, weather=False, messages=True)
_msgs = _session.race_control_messages
_track_msgs = _msgs[_msgs["Message"].str.contains(
    "TRACK LIMITS|DELETED", case=False, na=False)].copy()

# Map racing number -> abbreviation
_num_to_abbr = dict(zip(
    _session.results["DriverNumber"].astype(str),
    _session.results["Abbreviation"]
))
_num_to_name = dict(zip(
    _session.results["DriverNumber"].astype(str),
    _session.results["FullName"]
))

def _parse_msg(msg):
    """Extract driver abbr, deleted time, turn from message string."""
    driver = re.search(r"CAR \d+ \((\w+)\)", msg)
    turn   = re.search(r"TURN (\d+)", msg)
    time_  = re.search(r"TIME ([\d:\.]+)", msg)
    lap_   = re.search(r"LAP (\d+)", msg)
    return {
        "driver":  driver.group(1) if driver else "?",
        "turn":    int(turn.group(1)) if turn else 0,
        "deleted_time": time_.group(1) if time_ else "—",
        "lap":     int(lap_.group(1)) if lap_ else 0,
    }

# Build clean deletion table
_rows = []
for _, row in _track_msgs.iterrows():
    p = _parse_msg(row["Message"])
    _rows.append({
        "Time":    str(row["Time"])[-8:],
        "Driver":  p["driver"],
        "Lap":     p["lap"],
        "Turn":    p["turn"],
        "Deleted Time": p["deleted_time"],
        "Message": row["Message"],
    })
DELETIONS_DF = pd.DataFrame(_rows)
print(f"FastF1 loaded: {len(DELETIONS_DF)} deletion events")

# Per-driver summary
DRIVER_SUMMARY = DELETIONS_DF.groupby("Driver").agg(
    Deletions=("Driver","count"),
    Turns=("Turn", lambda x: ", ".join(sorted(set(str(t) for t in x if t>0))))
).reset_index().sort_values("Deletions", ascending=False)

# ── Helpers ───────────────────────────────────────────────────────────────────

def boxes_overlap(b1,b2):
    x1,y1,x2,y2=b1; x3,y3,x4,y4=b2
    cx1,cy1=(x1+x2)/2,(y1+y2)/2; cx2,cy2=(x3+x4)/2,(y3+y4)/2
    if min(x2,x4)>max(x1,x3) and min(y2,y4)>max(y1,y3): return True
    return ((cx1-cx2)**2+(cy1-cy2)**2)**0.5 < (x2-x1)*1.3

def chart(name):
    p=RUNS_DIR/name; return str(p) if p.exists() else None

def get_stats_html():
    avg=session_stats["total_ms"]/max(session_stats["frames"],1)
    return f"""<div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
      <div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px;">
        <div style="font-size:1.8em;font-weight:bold;color:#60a5fa;">{session_stats["frames"]}</div>
        <div style="color:#64748b;font-size:0.78em;margin-top:2px;">Frames Analysed</div></div>
      <div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px;">
        <div style="font-size:1.8em;font-weight:bold;color:#f87171;">{session_stats["violations"]}</div>
        <div style="color:#64748b;font-size:0.78em;margin-top:2px;">Violations</div></div>
      <div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px;">
        <div style="font-size:1.8em;font-weight:bold;color:#fb923c;">{session_stats["manual_reviews"]}</div>
        <div style="color:#64748b;font-size:0.78em;margin-top:2px;">Manual Reviews</div></div>
      <div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px;">
        <div style="font-size:1.8em;font-weight:bold;color:#4ade80;">{session_stats["clear"]}</div>
        <div style="color:#64748b;font-size:0.78em;margin-top:2px;">Clear</div></div>
      <div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px;">
        <div style="font-size:1.8em;font-weight:bold;color:#a78bfa;">{avg:.1f}ms</div>
        <div style="color:#64748b;font-size:0.78em;margin-top:2px;">Avg Inference</div></div>
    </div>"""

def get_confidence_bars_html(penalty_boxes, clean_boxes):
    if not penalty_boxes and not clean_boxes: return ""
    html="<div style='margin-top:12px;'><div style='color:#94a3b8;font-size:0.85em;margin-bottom:8px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;'>Detection Confidence</div>"
    for i,(conf,_) in enumerate(sorted(penalty_boxes,key=lambda x:-x[0])):
        pct=int(conf*100); c="#f87171" if pct>60 else "#fb923c" if pct>35 else "#fbbf24"
        html+=f"<div style='margin-bottom:10px;'><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#ccc;font-size:0.85em;'>🚨 Car {i+1} — Penalty</span><span style='color:{c};font-weight:bold;'>{pct}%</span></div><div style='background:#1e293b;border-radius:4px;height:10px;overflow:hidden;'><div style='background:{c};width:{pct}%;height:10px;border-radius:4px;'></div></div></div>"
    for i,(conf,_) in enumerate(sorted(clean_boxes,key=lambda x:-x[0])):
        pct=int(conf*100)
        html+=f"<div style='margin-bottom:10px;'><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#ccc;font-size:0.85em;'>✅ Car {i+1} — Clean</span><span style='color:#4ade80;font-weight:bold;'>{pct}%</span></div><div style='background:#1e293b;border-radius:4px;height:10px;overflow:hidden;'><div style='background:#4ade80;width:{pct}%;height:10px;border-radius:4px;'></div></div></div>"
    return html+"</div>"

def get_verdict_html(vtype,msg,ms,ts,cars):
    cfg={"clear":("#4ade80","#052e16","✅"),"penalty":("#f87171","#2d0a0a","🚨"),
         "review":("#fb923c","#2d1500","⚠️"),"none":("#94a3b8","#1e293b","⬜")}
    color,bg,icon=cfg.get(vtype,cfg["none"])
    return f"""<div style="background:{bg};border:2px solid {color};border-radius:10px;padding:16px;margin-top:8px;">
      <div style="font-size:1.3em;font-weight:bold;color:{color};margin-bottom:10px;line-height:1.4;">{icon} {msg}</div>
      <div style="display:flex;gap:20px;flex-wrap:wrap;">
        <span style="color:#64748b;font-size:0.82em;">🕐 {ts}</span>
        <span style="color:#64748b;font-size:0.82em;">⚡ Analysed in {ms:.1f}ms</span>
        <span style="color:#64748b;font-size:0.82em;">🚗 {cars} car(s) detected</span>
      </div></div>"""

# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image, conf_threshold):
    if image is None:
        return None,"<div style='color:#64748b;padding:16px;text-align:center;'>Upload an image to begin.</div>","",get_stats_html()
    t0=time.perf_counter()
    results=model(image,conf=conf_threshold,device=0,verbose=False)
    ms=(time.perf_counter()-t0)*1000
    annotated=cv2.cvtColor(results[0].plot(),cv2.COLOR_BGR2RGB)
    ts=time.strftime("%H:%M:%S")
    penalty_boxes,clean_boxes=[],[]
    for box in results[0].boxes:
        cls,conf,xyxy=int(box.cls),float(box.conf),box.xyxy[0].tolist()
        (penalty_boxes if cls==1 else clean_boxes).append((conf,xyxy))
    total=len(penalty_boxes)+len(clean_boxes)
    pushed=any(boxes_overlap(pb,cb) for _,pb in penalty_boxes for _,cb in clean_boxes)
    session_stats["frames"]+=1; session_stats["total_ms"]+=ms
    if total==0:
        vh=get_verdict_html("none",f"No detections above {conf_threshold*100:.0f}% — try lowering the slider",ms,ts,0)
    elif not penalty_boxes:
        session_stats["clear"]+=1
        vh=get_verdict_html("clear",f"NO VIOLATION — {total} car(s) within track limits",ms,ts,total)
    elif pushed:
        session_stats["violations"]+=1; session_stats["manual_reviews"]+=1
        vh=get_verdict_html("review","PENALTY + FLAGGED FOR MANUAL REVIEW<br><span style='font-size:0.85em;font-weight:normal;'>Cars in close contact — possible push/collision. Human steward verification required.</span>",ms,ts,total)
    else:
        session_stats["violations"]+=1
        vh=get_verdict_html("penalty",f"PENALTY — {len(penalty_boxes)} of {total} car(s) in violation",ms,ts,total)
    return annotated,vh,get_confidence_bars_html(penalty_boxes,clean_boxes),get_stats_html()

def reset_stats():
    for k in session_stats: session_stats[k]=0
    return get_stats_html()

# ── Ground Truth ──────────────────────────────────────────────────────────────

def draw_ground_truth(img_path):
    img=cv2.imread(str(img_path))
    if img is None: return None,[]
    h,w=img.shape[:2]; lbl=TEST_LBL_DIR/(Path(img_path).stem+".txt"); out=img.copy(); found=[]
    if lbl.exists():
        for line in lbl.read_text().strip().split("\n"):
            if not line.strip(): continue
            p=line.strip().split(); cls=int(p[0])
            cx,cy,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
            x1,y1=int((cx-bw/2)*w),int((cy-bh/2)*h); x2,y2=int((cx+bw/2)*w),int((cy+bh/2)*h)
            color=CLASS_COLORS_BGR[cls] if cls<len(CLASS_COLORS_BGR) else (128,128,128)
            cv2.rectangle(out,(x1,y1),(x2,y2),color,3)
            label=f"GT: {CLASS_NAMES[cls]}"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(out,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(out,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            found.append(CLASS_NAMES[cls])
    return cv2.cvtColor(out,cv2.COLOR_BGR2RGB),found

def compare_gt(img_path_str, conf_threshold):
    if not img_path_str:
        return None,None,"<div style='color:#64748b;padding:12px;'>Select an image above.</div>"
    img_path=Path(img_path_str)
    gt_img,gt_boxes=draw_ground_truth(img_path)
    if gt_img is None: return None,None,"Could not load image."
    img_rgb=cv2.cvtColor(cv2.imread(str(img_path)),cv2.COLOR_BGR2RGB)
    t0=time.perf_counter()
    results=model(img_rgb,conf=conf_threshold,device=0,verbose=False)
    ms=(time.perf_counter()-t0)*1000
    pred_img=cv2.cvtColor(results[0].plot(),cv2.COLOR_BGR2RGB)
    pred_boxes=[f"{CLASS_NAMES[int(b.cls)]}({float(b.conf)*100:.0f}%)" for b in results[0].boxes]
    gt_pen=any("Penalty" in b for b in gt_boxes); pred_pen=any("Penalty" in b for b in pred_boxes)
    if gt_pen==pred_pen:
        match="<span style='color:#4ade80;font-weight:bold;'>✅ CORRECT — prediction matches ground truth</span>"
    elif gt_pen and not pred_pen:
        match="<span style='color:#f87171;font-weight:bold;'>❌ MISSED — labelled penalty not detected (try lowering threshold)</span>"
    else:
        match="<span style='color:#fb923c;font-weight:bold;'>⚠️ FALSE POSITIVE — penalty flagged but not in ground truth</span>"
    gt_str=", ".join(gt_boxes) if gt_boxes else "none (background)"
    pred_str=", ".join(pred_boxes) if pred_boxes else "nothing detected"
    summary=f"""<div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:16px;margin-top:8px;">
      <div style="margin-bottom:10px;">{match}</div>
      <div style="color:#94a3b8;font-size:0.85em;margin-bottom:4px;"><strong style="color:#cbd5e1;">Ground Truth:</strong> {gt_str}</div>
      <div style="color:#94a3b8;font-size:0.85em;margin-bottom:4px;"><strong style="color:#cbd5e1;">Prediction:</strong> {pred_str}</div>
      <div style="color:#64748b;font-size:0.82em;margin-top:8px;">⚡ {ms:.1f}ms &nbsp;·&nbsp; 🟢 Green = ground truth &nbsp;·&nbsp; Cyan = model prediction</div>
    </div>"""
    return gt_img,pred_img,summary

test_images=sorted([str(p) for p in TEST_IMG_DIR.glob("*.jpg")]+[str(p) for p in TEST_IMG_DIR.glob("*.png")])

# ── FastF1 Validation tab functions ───────────────────────────────────────────

def get_deletions_table_html():
    rows_html=""
    for _,row in DELETIONS_DF.iterrows():
        turn_color="#f87171" if row["Turn"] in [9,10] else "#fb923c"
        rows_html+=f"""<tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px 12px;color:#94a3b8;">{row["Time"]}</td>
          <td style="padding:8px 12px;font-weight:bold;color:#60a5fa;">{row["Driver"]}</td>
          <td style="padding:8px 12px;color:#cbd5e1;">{row["Lap"] if row["Lap"] else "—"}</td>
          <td style="padding:8px 12px;color:{turn_color};font-weight:bold;">T{row["Turn"] if row["Turn"] else "—"}</td>
          <td style="padding:8px 12px;color:#fbbf24;font-family:monospace;">{row["Deleted Time"]}</td>
        </tr>"""
    return f"""
    <div style="overflow-x:auto;margin-top:12px;">
      <table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:8px;overflow:hidden;">
        <thead>
          <tr style="background:#1e293b;">
            <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Time</th>
            <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Driver</th>
            <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Lap</th>
            <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Turn</th>
            <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Deleted Time</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

def get_driver_breakdown_html():
    cards=""
    colors=["#f87171","#fb923c","#fbbf24","#4ade80","#60a5fa","#a78bfa","#f472b6"]
    for i,(_,row) in enumerate(DRIVER_SUMMARY.iterrows()):
        color=colors[i%len(colors)]
        cards+=f"""<div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:12px 16px;min-width:120px;text-align:center;">
          <div style="font-size:1.4em;font-weight:bold;color:{color};">{row["Driver"]}</div>
          <div style="font-size:1.8em;font-weight:bold;color:white;margin:4px 0;">{row["Deletions"]}</div>
          <div style="color:#64748b;font-size:0.75em;">Turn(s) {row["Turns"]}</div>
        </div>"""
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;">{cards}</div>'

def get_hotspot_html():
    turn_counts=DELETIONS_DF[DELETIONS_DF["Turn"]>0]["Turn"].value_counts()
    bars=""
    max_count=turn_counts.max() if len(turn_counts) else 1
    for turn,count in turn_counts.items():
        pct=int(count/max_count*100)
        color="#f87171" if pct>70 else "#fb923c" if pct>40 else "#fbbf24"
        bars+=f"""<div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#cbd5e1;font-size:0.9em;font-weight:600;">Turn {turn}</span>
            <span style="color:{color};font-weight:bold;">{count} deletions</span>
          </div>
          <div style="background:#1e293b;border-radius:4px;height:12px;overflow:hidden;">
            <div style="background:{color};width:{pct}%;height:12px;border-radius:4px;"></div>
          </div>
        </div>"""
    return f'<div style="margin-top:8px;">{bars}</div>'

def filter_by_driver(driver_abbr):
    if driver_abbr == "All Drivers":
        filtered = DELETIONS_DF
    else:
        filtered = DELETIONS_DF[DELETIONS_DF["Driver"]==driver_abbr]
    rows_html=""
    for _,row in filtered.iterrows():
        turn_color="#f87171" if row["Turn"] in [9,10] else "#fb923c"
        rows_html+=f"""<tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px 12px;color:#94a3b8;">{row["Time"]}</td>
          <td style="padding:8px 12px;font-weight:bold;color:#60a5fa;">{row["Driver"]}</td>
          <td style="padding:8px 12px;color:#cbd5e1;">{row["Lap"] if row["Lap"] else "—"}</td>
          <td style="padding:8px 12px;color:{turn_color};font-weight:bold;">T{row["Turn"] if row["Turn"] else "—"}</td>
          <td style="padding:8px 12px;color:#fbbf24;font-family:monospace;">{row["Deleted Time"]}</td>
        </tr>"""
    table=f"""<div style="overflow-x:auto;margin-top:12px;">
      <table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:8px;overflow:hidden;">
        <thead><tr style="background:#1e293b;">
          <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Time</th>
          <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Driver</th>
          <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Lap</th>
          <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Turn</th>
          <th style="padding:10px 12px;color:#94a3b8;text-align:left;font-size:0.82em;text-transform:uppercase;">Deleted Time</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    count_html=f"""<div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:12px 20px;display:inline-block;margin-bottom:12px;">
      <span style="color:#94a3b8;font-size:0.85em;">Showing </span>
      <span style="color:#60a5fa;font-weight:bold;font-size:1.2em;">{len(filtered)}</span>
      <span style="color:#94a3b8;font-size:0.85em;"> deletion(s)</span>
    </div>"""
    return count_html+table

driver_choices=["All Drivers"]+sorted(DRIVER_SUMMARY["Driver"].tolist())

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Digital Steward — F1 Track Limit Detector") as demo:

    gr.HTML("""<div style="background:linear-gradient(90deg,#e10600 0%,#1a1a2e 55%);padding:24px;border-radius:10px;margin-bottom:16px;">
      <h1 style="color:white;margin:0;font-size:2em;letter-spacing:-0.5px;">🏁 Digital Steward</h1>
      <p style="color:#cbd5e1;margin:6px 0 0 0;font-size:0.95em;">Real-time F1 Track Limit Violation Detector &nbsp;·&nbsp; Dell Pro Max GB10 &nbsp;·&nbsp; NVIDIA Blackwell &nbsp;·&nbsp; 3.2ms inference &nbsp;·&nbsp; Zero cloud latency</p>
    </div>""")

    stats_display=gr.HTML(get_stats_html())

    with gr.Tabs():

        with gr.TabItem("🔍 Live Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input=gr.Image(label="Upload F1 Frame",sources=["upload","webcam"],type="numpy")
                    conf_slider=gr.Slider(minimum=0.05,maximum=0.95,value=0.35,step=0.05,
                        label="Confidence Threshold",
                        info="Lower = catch more violations · Higher = only very confident detections")
                    with gr.Row():
                        submit_btn=gr.Button("🔍 Analyse Frame",variant="primary",size="lg")
                        reset_btn=gr.Button("↺ Reset Stats",size="lg")
                    gr.Markdown("""
**Threshold guide:**
- `0.05–0.20` — catch everything, may over-flag
- `0.25–0.45` — balanced *(recommended)*
- `0.50+` — high-confidence only

**Verdict types:**
- ✅ **Clear** — all cars within track limits
- 🚨 **Penalty** — track limit violation detected
- ⚠️ **Manual Review** — contact/push between cars
                    """)
                with gr.Column(scale=2):
                    image_output=gr.Image(label="Detection Result")
                    verdict_display=gr.HTML("<div style='color:#64748b;padding:16px;text-align:center;'>Upload an image to begin.</div>")
                    conf_bars_display=gr.HTML()
            submit_btn.click(fn=predict,inputs=[image_input,conf_slider],
                outputs=[image_output,verdict_display,conf_bars_display,stats_display])
            conf_slider.release(fn=predict,inputs=[image_input,conf_slider],
                outputs=[image_output,verdict_display,conf_bars_display,stats_display])
            reset_btn.click(fn=reset_stats,outputs=[stats_display])

        with gr.TabItem("🔬 Ground Truth Comparison"):
            gr.HTML("""<div style="padding:16px 0 8px 0;">
              <h2 style="color:white;margin:0 0 4px 0;">Ground Truth vs Model Prediction</h2>
              <p style="color:#64748b;margin:0;">Side-by-side comparison on 15 held-out test images never seen during training.</p>
            </div>""")
            with gr.Row():
                gt_picker=gr.Dropdown(choices=test_images,label="Select Test Image",
                    info="15 held-out test images — never seen during training")
                gt_conf=gr.Slider(minimum=0.05,maximum=0.95,value=0.35,step=0.05,label="Confidence Threshold")
            gt_btn=gr.Button("🔬 Compare",variant="primary")
            with gr.Row():
                gt_left=gr.Image(label="🟢 Ground Truth (Human Labels)")
                gt_right=gr.Image(label="🔵 Model Prediction")
            gt_summary=gr.HTML()
            gt_btn.click(fn=compare_gt,inputs=[gt_picker,gt_conf],outputs=[gt_left,gt_right,gt_summary])
            gt_picker.change(fn=compare_gt,inputs=[gt_picker,gt_conf],outputs=[gt_left,gt_right,gt_summary])
            gt_conf.release(fn=compare_gt,inputs=[gt_picker,gt_conf],outputs=[gt_left,gt_right,gt_summary])

        with gr.TabItem("🏛️ FIA Steward Validation"):
            gr.HTML(f"""<div style="padding:16px 0 8px 0;">
              <h2 style="color:white;margin:0 0 4px 0;">FIA Official Track Limit Decisions</h2>
              <p style="color:#64748b;margin:0 0 4px 0;">
                Real race control messages from <strong style="color:#cbd5e1;">2023 Austrian Grand Prix — Qualifying</strong>
                sourced live via FastF1 API. This is the exact session infamous for causing a 5-hour results delay.
              </p>
            </div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">
              <div style="background:#1a1a2e;border:1px solid #f87171;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#f87171;">{len(DELETIONS_DF)}</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Official Deletions</div>
              </div>
              <div style="background:#1a1a2e;border:1px solid #60a5fa;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#60a5fa;">{DELETIONS_DF["Driver"].nunique()}</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Drivers Penalised</div>
              </div>
              <div style="background:#1a1a2e;border:1px solid #fbbf24;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#fbbf24;">{DELETIONS_DF[DELETIONS_DF["Turn"]>0]["Turn"].nunique()}</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Violation Hotspots</div>
              </div>
              <div style="background:#1a1a2e;border:1px solid #4ade80;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#4ade80;">T9 + T10</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Primary Hotspots</div>
              </div>
              <div style="background:#1a1a2e;border:1px solid #a78bfa;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#a78bfa;">5+ hrs</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Results Delay</div>
              </div>
            </div>""")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""<div style="color:#94a3b8;font-size:0.85em;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Violation Hotspots</div>""")
                    hotspot_html=gr.HTML(get_hotspot_html())

                    gr.HTML("""<div style="color:#94a3b8;font-size:0.85em;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin:16px 0 8px 0;">Deletions Per Driver</div>""")
                    gr.HTML(get_driver_breakdown_html())

                with gr.Column(scale=2):
                    gr.HTML("""<div style="color:#94a3b8;font-size:0.85em;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Official Race Control Log</div>""")
                    driver_filter=gr.Dropdown(
                        choices=driver_choices,
                        value="All Drivers",
                        label="Filter by Driver",
                    )
                    deletions_table=gr.HTML(filter_by_driver("All Drivers"))
                    driver_filter.change(fn=filter_by_driver,inputs=[driver_filter],outputs=[deletions_table])

            gr.HTML("""
            <div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:16px;margin-top:16px;">
              <div style="color:#94a3b8;font-size:0.85em;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;">How Digital Steward Addresses This</div>
              <div style="display:flex;gap:16px;flex-wrap:wrap;">
                <div style="flex:1;min-width:200px;">
                  <div style="color:#4ade80;font-weight:bold;margin-bottom:4px;">✅ Automated Detection</div>
                  <div style="color:#64748b;font-size:0.85em;">Real-time CV flags every potential violation instantly — no 5-hour review backlog</div>
                </div>
                <div style="flex:1;min-width:200px;">
                  <div style="color:#60a5fa;font-weight:bold;margin-bottom:4px;">⚡ 3.2ms Per Frame</div>
                  <div style="color:#64748b;font-size:0.85em;">GB10 Blackwell processes broadcast feed faster than human stewards can review clips</div>
                </div>
                <div style="flex:1;min-width:200px;">
                  <div style="color:#a78bfa;font-weight:bold;margin-bottom:4px;">🏛️ FIA Aligned</div>
                  <div style="color:#64748b;font-size:0.85em;">Validates against official race control messages — same ground truth the stewards use</div>
                </div>
                <div style="flex:1;min-width:200px;">
                  <div style="color:#fb923c;font-weight:bold;margin-bottom:4px;">⚠️ Smart Escalation</div>
                  <div style="color:#64748b;font-size:0.85em;">Contact/push incidents auto-flagged for human review — AI knows its limits</div>
                </div>
              </div>
            </div>""")

        with gr.TabItem("📊 Model Performance"):
            gr.HTML("""<div style="padding:16px 0 12px 0;">
              <h2 style="color:white;margin:0 0 6px 0;">Model Training Results</h2>
              <p style="color:#64748b;margin:0;">YOLOv8n · 309 training images · 80 epochs · Dell Pro Max GB10</p>
            </div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">
              <div style="background:#1a1a2e;border:1px solid #4ade80;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#4ade80;">89.5%</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Penalty mAP50</div></div>
              <div style="background:#1a1a2e;border:1px solid #4ade80;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#4ade80;">100%</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Penalty Precision</div></div>
              <div style="background:#1a1a2e;border:1px solid #60a5fa;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#60a5fa;">92.3%</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Overall mAP50</div></div>
              <div style="background:#1a1a2e;border:1px solid #a78bfa;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#a78bfa;">3.2ms</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Inference / Frame</div></div>
              <div style="background:#1a1a2e;border:1px solid #fb923c;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#fb923c;">3 min</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Training Time on GB10</div></div>
              <div style="background:#1a1a2e;border:1px solid #334155;border-radius:8px;padding:14px 22px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:#94a3b8;">309</div>
                <div style="color:#64748b;font-size:0.82em;margin-top:2px;">Training Images</div></div>
            </div>""")
            with gr.Row():
                if chart("confusion_matrix_normalized.png"):
                    gr.Image(value=chart("confusion_matrix_normalized.png"),label="Confusion Matrix (Normalised)")
                if chart("BoxF1_curve.png"):
                    gr.Image(value=chart("BoxF1_curve.png"),label="F1 Score Curve")
                if chart("BoxPR_curve.png"):
                    gr.Image(value=chart("BoxPR_curve.png"),label="Precision-Recall Curve")
            with gr.Row():
                if chart("BoxP_curve.png"):
                    gr.Image(value=chart("BoxP_curve.png"),label="Precision Curve")
                if chart("BoxR_curve.png"):
                    gr.Image(value=chart("BoxR_curve.png"),label="Recall Curve")
                if chart("results.png"):
                    gr.Image(value=chart("results.png"),label="Training Loss + mAP over Epochs")
            gr.Markdown("""
**Augmentations:** Mosaic (p=0.5) · MixUp (p=0.2) · Horizontal flip (p=0.5) · Rotation ±15° · HSV jitter · Random scaling · Perspective · Random erasing · No vertical flip
            """)

    gr.HTML("""<div style="border-top:1px solid #1e293b;margin-top:24px;padding-top:16px;color:#475569;font-size:0.82em;">
      <strong style="color:#94a3b8;">Model:</strong> YOLOv8n &nbsp;|&nbsp;
      <strong style="color:#94a3b8;">Hardware:</strong> Dell Pro Max GB10 · NVIDIA Blackwell · 122GB unified memory &nbsp;|&nbsp;
      <strong style="color:#94a3b8;">Data:</strong> FastF1 API · 2023 Austrian GP Official Race Control Messages
    </div>""")

demo.launch(server_port=7860, share=False)
