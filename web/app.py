"""Web UI (Gradio). Run from the project folder:  python web/app.py  then open http://localhost:7860"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowd.analyzer import DENSITY, DETECTION, Analyzer  # noqa: E402
from crowd.config import load_config  # noqa: E402
from crowd.media import open_video, read_first_frame, read_image  # noqa: E402
from crowd.render import zone_color  # noqa: E402
from crowd.zones import clamp_polygon, label_point, load_zones, save_zones  # noqa: E402

DEMO_DIR = ROOT / "data" / "demo"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
DISPLAY_WIDTH = 960      # frames are shrunk to this width before being sent to the browser
VIDEO_UPDATE_SEC = 0.1   # at most ~10 browser updates per second while a video plays
LEVEL_COLORS = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}

_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer(load_config())
    return _analyzer


def demo_files():
    files = (p for p in DEMO_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
    return sorted(p.relative_to(DEMO_DIR).as_posix() for p in files)


def selected_path(source, demo_choice, upload):
    if source == "Upload":
        return upload or None
    return str(DEMO_DIR / demo_choice) if demo_choice else None


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXT


def for_display(img_bgr):
    """BGR -> RGB, shrunk to DISPLAY_WIDTH."""
    h, w = img_bgr.shape[:2]
    if w > DISPLAY_WIDTH:
        img_bgr = cv2.resize(img_bgr, (DISPLAY_WIDTH, round(h * DISPLAY_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# --- Analyze tab

def stats_html(result):
    color = LEVEL_COLORS[result["level"]]
    note = "" if result["heatmap"] is not None else '<div class="note">Heatmap is available in Density mode.</div>'
    return (f'<div class="stat-card" style="border-color:{color}">'
            f'<div class="stat-label">People</div><div class="stat-value">{result["count"]}</div>'
            f'<div class="stat-level" style="background:{color}">{result["level"]}</div>{note}</div>')


def outputs_for(result):
    heatmap = for_display(result["heatmap"]) if result["heatmap"] is not None else None
    table = [[name, count] for name, count in result["zone_counts"]]
    return for_display(result["overlay"]), heatmap, stats_html(result), table


def analyze(source, demo_choice, upload, mode_label):
    """Yield (frame, heatmap, stats html, zone table) for an image once, or for a video repeatedly."""
    path = selected_path(source, demo_choice, upload)
    if not path:
        raise gr.Error("Choose a demo file or upload an image/video first.")
    mode = DETECTION if mode_label == "Detection" else DENSITY
    if mode == DENSITY and not get_analyzer().density_available:
        raise gr.Error("Density model is not installed. Use Detection mode, "
                       "or add weights/crowd_vgg19.pth (see README).")
    zones = load_zones(path)
    if is_image(path):
        image = read_image(path)
        if image is None:
            raise gr.Error("Cannot read this image.")
        yield outputs_for(get_analyzer().process(image, mode, zones))
        return
    cap = open_video(path)
    if cap is None:
        raise gr.Error("Cannot open this video. Tip: avoid non-English characters in the file path.")
    last_sent = 0.0
    try:
        for result in get_analyzer().stream(cap, mode, zones):
            now = time.perf_counter()
            if result["fresh"] or now - last_sent >= VIDEO_UPDATE_SEC:
                last_sent = now
                yield outputs_for(result)
    finally:
        cap.release()


# --- Zones tab (state: path, frame, saved zones, points of the polygon being drawn)

def empty_zone_state():
    return {"path": None, "frame": None, "zones": [], "points": []}


def render_zones(state):
    frame = state["frame"]
    if frame is None:
        return None
    out = frame.copy()
    fill = frame.copy()
    thick = max(2, frame.shape[1] // 640)
    for i, zone in enumerate(state["zones"]):
        pts = np.array(zone["points"], np.int32)
        cv2.fillPoly(fill, [pts], zone_color(i))
    out = cv2.addWeighted(fill, 0.25, out, 0.75, 0)
    for i, zone in enumerate(state["zones"]):
        pts = np.array(zone["points"], np.int32)
        cv2.polylines(out, [pts], True, zone_color(i), thick)
        x, y = label_point(zone["points"])
        cv2.putText(out, zone["name"], (x - 30, y), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, frame.shape[1] / 1400),
                    zone_color(i), 2, cv2.LINE_AA)
    points = state["points"]
    if len(points) > 1:
        cv2.polylines(out, [np.array(points, np.int32)], False, (0, 0, 255), thick)
    for p in points:
        cv2.circle(out, tuple(p), thick + 4, (0, 0, 255), -1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def zone_outputs(state, status=""):
    names = [z["name"] for z in state["zones"]]
    table = [[z["name"], len(z["points"])] for z in state["zones"]]
    if not status and state["frame"] is not None:
        status = f"{len(state['points'])} point(s) in the current polygon."
    return state, render_zones(state), table, gr.update(choices=names, value=None), status


def zones_load(source, demo_choice, upload):
    path = selected_path(source, demo_choice, upload)
    if not path:
        raise gr.Error("Choose a demo file or upload an image/video first.")
    frame = read_image(path) if is_image(path) else read_first_frame(path)
    if frame is None:
        raise gr.Error("Cannot read this file.")
    state = {"path": path, "frame": frame, "zones": load_zones(path), "points": []}
    return zone_outputs(state, f"Loaded {Path(path).name} with {len(state['zones'])} zone(s).")


def zones_click(state, evt: gr.SelectData):
    if state["frame"] is not None:
        state["points"].append((int(evt.index[0]), int(evt.index[1])))
    return zone_outputs(state)


def zones_undo(state):
    if state["points"]:
        state["points"].pop()
    return zone_outputs(state)


def zones_clear(state):
    state["points"] = []
    return zone_outputs(state)


def zones_add(state, name):
    if state["frame"] is None:
        return zone_outputs(state)
    h, w = state["frame"].shape[:2]
    points = clamp_polygon(state["points"], w, h)
    if points is None:
        return zone_outputs(state, "A zone needs at least 3 points that enclose an area.")
    used = {z["name"] for z in state["zones"]}
    name = (name or "").strip()
    if not name:
        i = 1
        while f"Zone {i}" in used:
            i += 1
        name = f"Zone {i}"
    state["zones"].append({"name": name, "points": points})
    state["points"] = []
    return zone_outputs(state, f"Added {name}. Click Save to keep it.")


def zones_delete(state, name):
    state["zones"] = [z for z in state["zones"] if z["name"] != name]
    return zone_outputs(state, f"Deleted {name}." if name else "")


def zones_save(state):
    if state["path"] is None:
        return zone_outputs(state)
    path = save_zones(state["path"], state["zones"])
    return zone_outputs(state, f"Saved {len(state['zones'])} zone(s) to {path.name}.")


# --- Layout

CSS = """
.stat-card {border-left: 8px solid; border-radius: 12px; padding: 14px 18px; background: var(--block-background-fill);}
.stat-label {font-size: 0.9rem; opacity: 0.7;}
.stat-value {font-size: 2.6rem; font-weight: 700; line-height: 1.1;}
.stat-level {display: inline-block; color: white; font-weight: 600; padding: 2px 12px; border-radius: 999px; margin-top: 6px;}
.note {margin-top: 8px; font-size: 0.85rem; opacity: 0.7;}
"""


def build_ui():
    files = demo_files()
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(title="AI Crowd Analytics", theme=theme, css=CSS) as ui:
        gr.Markdown("# AI Crowd Analytics\nEstimate crowd size and density in images and videos. "
                    "**Density** mode counts dense crowds and shows a heatmap; **Detection** mode draws a box "
                    "around each person (best for sparse scenes).")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                source = gr.Radio(["Demo file", "Upload"], value="Demo file", label="Source")
                demo_choice = gr.Dropdown(files, value=files[0] if files else None, label="Demo file")
                upload = gr.File(label="Upload image or video", type="filepath",
                                 file_types=sorted(IMAGE_EXT | VIDEO_EXT))
                density_ok = get_analyzer().density_available
                mode = gr.Radio(["Density", "Detection"], value="Density" if density_ok else "Detection", label="Mode",
                                info=None if density_ok else "Density model not installed - Detection mode only")
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")
                stats = gr.HTML()
                zone_table = gr.Dataframe(headers=["Zone", "Count"], label="People per zone", interactive=False)
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Analyze"):
                        with gr.Row():
                            frame_view = gr.Image(label="Analysis", format="jpeg", interactive=False)
                            heatmap_view = gr.Image(label="Density heatmap", format="jpeg", interactive=False)
                    with gr.Tab("Zones"):
                        gr.Markdown("1. Choose a file on the left and click **Load**. "
                                    "2. Click on the image to add corner points. "
                                    "3. Type a name (optional) and click **Add zone**. "
                                    "4. Click **Save** - the zones are used when you run this file.")
                        zone_state = gr.State(empty_zone_state())
                        canvas = gr.Image(label="Click to add points", interactive=False)
                        with gr.Row():
                            load_btn = gr.Button("Load")
                            undo_btn = gr.Button("Undo point")
                            clear_btn = gr.Button("Clear points")
                        with gr.Row():
                            zone_name = gr.Textbox(label="Zone name", placeholder="e.g. Entrance")
                            add_btn = gr.Button("Add zone", variant="primary")
                        with gr.Row():
                            delete_choice = gr.Dropdown([], label="Zone to delete")
                            delete_btn = gr.Button("Delete zone")
                            save_btn = gr.Button("Save", variant="primary")
                        saved_table = gr.Dataframe(headers=["Zone", "Points"], label="Zones", interactive=False)
                        zone_status = gr.Markdown()

        run_event = run_btn.click(analyze, [source, demo_choice, upload, mode],
                                  [frame_view, heatmap_view, stats, zone_table])
        stop_btn.click(None, cancels=[run_event])

        zone_outputs_list = [zone_state, canvas, saved_table, delete_choice, zone_status]
        load_btn.click(zones_load, [source, demo_choice, upload], zone_outputs_list)
        canvas.select(zones_click, [zone_state], zone_outputs_list)
        undo_btn.click(zones_undo, [zone_state], zone_outputs_list)
        clear_btn.click(zones_clear, [zone_state], zone_outputs_list)
        add_btn.click(zones_add, [zone_state, zone_name], zone_outputs_list)
        delete_btn.click(zones_delete, [zone_state, delete_choice], zone_outputs_list)
        save_btn.click(zones_save, [zone_state], zone_outputs_list)
    return ui


if __name__ == "__main__":
    if get_analyzer().density_available:
        get_analyzer().density  # load the model before the first click
    build_ui().queue().launch(inbrowser=True)
