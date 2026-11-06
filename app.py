from pathlib import Path
from collections import Counter
import io
import threading
import time
import queue

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    enumerate_cameras = None

st.set_page_config(page_title="PCB Defect Detection", page_icon="🔬", layout="wide")
st.title("🔬 PCB Defect Detection")
st.caption("AI-assisted PCB inspection with image upload, built-in webcam, USB webcam, or DroidCam")

PROJECT_DIR = Path(__file__).resolve().parent
def resolve_default_weight(relative_path):
    """Resolve legacy result paths relative to this project, never a user's drive."""
    return str(PROJECT_DIR / relative_path)

MODEL_PATHS = {
    "PCB Defect Model (6 classes)": str(
        PROJECT_DIR / "models" / "pcb_yolo_baseline_w0.pt"
    ),
    "DeepPCB Model (6 classes)": str(
        PROJECT_DIR / "models" / "deeppcb_yolo_baseline.pt"
    )
}

st.sidebar.header("Inspection settings")
model_choice = st.sidebar.selectbox("Choose trained model", list(MODEL_PATHS))
selected_weights_path = Path(MODEL_PATHS[model_choice])
st.sidebar.caption(f"Model file: models/{selected_weights_path.name}")
if not selected_weights_path.is_file():
    st.sidebar.error(
        f"Model file not found: models/{selected_weights_path.name}. "
        "Make sure the models folder and selected .pt file are included in the project."
    )
confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
imgsz = st.sidebar.select_slider("Inference image size", [320, 416, 512, 608, 640, 768, 960], value=608)
st.sidebar.info(
    "Live-camera inspection uses only the rectangular ROI (region of interest). "
    "It does not use overlapping tiles or segmentation. Predictions are not proof of a real defect."
)

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(path):
    return YOLO(path)

@st.cache_data(ttl=15, show_spinner=False)
def discover_physical_cameras():
    """Enumerate cameras per backend and merge duplicate backend entries."""
    if enumerate_cameras is None:
        return []

    discovered = {}
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    if not backends:
        backends.append(cv2.CAP_ANY)

    for backend_id in backends:
        try:
            camera_list = enumerate_cameras(backend_id)
        except Exception:
            continue
        for camera in camera_list:
            name = str(getattr(camera, "name", "Camera")).strip() or "Camera"
            path = str(getattr(camera, "path", "") or "")
            vid = getattr(camera, "vid", None)
            pid = getattr(camera, "pid", None)
            index = int(camera.index)
            actual_backend = int(getattr(camera, "backend", backend_id) or backend_id)

            # Windows assigns a different interface GUID per backend. The stable
            # device-instance portion before that GUID identifies the same camera.
            stable_path = path.split("#{", 1)[0].casefold() if path else ""
            identity = (stable_path,) if stable_path else (name.casefold(), vid, pid)
            entry = discovered.setdefault(identity, {
                "name": name, "vid": vid, "pid": pid, "path": path,
                "captures": [],
            })
            capture = {"index": index, "backend": actual_backend}
            if capture not in entry["captures"]:
                entry["captures"].append(capture)

    devices = list(discovered.values())
    for device in devices:
        device["captures"].sort(key=lambda item: item["backend"] == cv2.CAP_DSHOW, reverse=True)
        name = device["name"].casefold()
        # Classification is best-effort: Windows does not expose a universal
        # built-in-vs-external flag for every webcam.
        laptop_hints = ("integrated", "built-in", "builtin", "laptop", "wide vision", "facetime", "internal")
        virtual_hints = ("droidcam", "dev47", "obs virtual", "virtual camera")
        if any(hint in name for hint in virtual_hints):
            device["category"] = "Virtual"
        elif any(hint in name for hint in laptop_hints):
            device["category"] = "Laptop Camera"
        else:
            device["category"] = "USB Camera"

    return sorted(devices, key=lambda item: (item["category"], item["name"].casefold()))


def open_camera(source_type, camera_index=None, backend=None, stream_url=None):
    """Open one camera capture. Each Streamlit session owns its own capture."""
    if source_type == "DroidCam":
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(stream_url)
    else:
        cap = cv2.VideoCapture(int(camera_index), int(backend))

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    cap.release()
    return None


class CameraReader:
    """Continuously capture frames in the background and expose only the newest one."""
    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if ok and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.02)

    def read_latest(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        self.capture.release()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class InferenceWorker:
    """Run YOLO away from the Streamlit preview loop."""
    def __init__(self):
        self.tasks = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.running = True
        self.busy = False
        self.latest_result = None
        self.latest_error = None
        self.thread = threading.Thread(target=self._work_loop, daemon=True)
        self.thread.start()

    def submit(self, roi_rgb, model, confidence, imgsz):
        with self.lock:
            if self.busy or not self.running:
                return False
            self.busy = True
            self.latest_error = None
        self.tasks.put((roi_rgb.copy(), model, confidence, imgsz))
        return True

    def _work_loop(self):
        while self.running:
            try:
                roi_rgb, model, confidence, imgsz = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
                result = model.predict(
                    source=roi_bgr, conf=confidence, imgsz=imgsz, verbose=False
                )[0]
                if result.boxes is None or len(result.boxes) == 0:
                    boxes, scores, classes = [], [], []
                else:
                    boxes = result.boxes.xyxy.cpu().numpy().tolist()
                    scores = result.boxes.conf.cpu().numpy().tolist()
                    classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
                annotated_roi = draw_predictions(
                    roi_rgb, boxes, scores, classes, model.names
                )
                output = {
                    "image": annotated_roi, "boxes": boxes, "scores": scores,
                    "classes": classes, "names": model.names,
                    "roi_size": (roi_rgb.shape[1], roi_rgb.shape[0]),
                }
                with self.lock:
                    self.latest_result = output
                    self.latest_error = None
            except Exception as exc:
                with self.lock:
                    self.latest_error = str(exc)
            finally:
                with self.lock:
                    self.busy = False
                self.tasks.task_done()

    def snapshot(self):
        with self.lock:
            return self.latest_result, self.latest_error, self.busy

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

def starts_for(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))

def predict_full(model, image, conf, size):
    # The UI stores images as RGB; Ultralytics expects OpenCV-style BGR arrays.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = model.predict(source=image_bgr, conf=conf, imgsz=size, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], [], [], 1, 0
    return (
        result.boxes.xyxy.cpu().numpy().tolist(),
        result.boxes.conf.cpu().numpy().tolist(),
        result.boxes.cls.cpu().numpy().astype(int).tolist(),
        1,
        len(result.boxes),
    )

def predict_tiles(model, image, conf, size, tw, th, overlap, nms_iou):
    h, w = image.shape[:2]
    sx, sy = max(1, int(tw * (1 - overlap))), max(1, int(th * (1 - overlap)))
    xs, ys = starts_for(w, tw, sx), starts_for(h, th, sy)
    boxes_all, scores_all, classes_all = [], [], []
    tiles = 0

    for y in ys:
        for x in xs:
            x2, y2 = min(w, x + tw), min(h, y + th)
            tile = image[y:y2, x:x2]
            if tile.size == 0:
                continue
            tiles += 1
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            result = model.predict(source=tile_bgr, conf=conf, imgsz=size, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            boxes_all.extend(boxes.tolist())
            scores_all.extend(result.boxes.conf.cpu().numpy().tolist())
            classes_all.extend(result.boxes.cls.cpu().numpy().astype(int).tolist())

    raw_count = len(boxes_all)
    if raw_count == 0:
        return [], [], [], tiles, 0

    boxes_all = np.asarray(boxes_all, dtype=np.float32)
    scores_all = np.asarray(scores_all, dtype=np.float32)
    classes_all = np.asarray(classes_all, dtype=np.int32)
    final_boxes, final_scores, final_classes = [], [], []

    for class_id in np.unique(classes_all):
        idx = np.where(classes_all == class_id)[0]
        xywh = [
            [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            for b in boxes_all[idx]
        ]
        keep = cv2.dnn.NMSBoxes(
            xywh, scores_all[idx].astype(float).tolist(), float(conf), float(nms_iou)
        )
        if len(keep):
            chosen = idx[np.asarray(keep).reshape(-1)]
            final_boxes.extend(boxes_all[chosen].tolist())
            final_scores.extend(scores_all[chosen].tolist())
            final_classes.extend(classes_all[chosen].tolist())

    return final_boxes, final_scores, final_classes, tiles, raw_count

def draw_predictions(image, boxes, scores, classes, names):
    out = image.copy()
    h, w = out.shape[:2]
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        color = (255, 40, 180)
        label = f"{names.get(int(class_id), class_id)} {float(score):.2f}"
        font_scale = max(0.45, min(0.75, w / 1400))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ly = max(th + 6, y1)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, max(0, ly - th - 6)), (min(w - 1, x1 + tw + 6), min(h - 1, ly + baseline)), color, -1)
        cv2.putText(out, label, (x1 + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def show_summary(boxes, scores, classes, names):
    c1, c2 = st.columns(2)
    c1.metric("Detections", len(boxes))
    c2.metric("Mean confidence", f"{np.mean(scores):.2f}" if scores else "N/A")
    if boxes:
        counts = Counter(names.get(int(c), str(c)) for c in classes)
        rows = []
        for class_name, count in counts.most_common():
            vals = [float(s) for s, c in zip(scores, classes) if names.get(int(c), str(c)) == class_name]
            rows.append({"Predicted class": class_name, "Count": count, "Mean confidence": round(float(np.mean(vals)), 3)})
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No detections returned at this confidence threshold.")

# Keep the two input paths separate: upload supports full-image or tiled inference;
# live camera always sends only the selected ROI crop to YOLO.
upload_tab, camera_tab = st.tabs(["📁 Upload image", "📷 Live camera (Webcam / DroidCam)"])

with upload_tab:
    st.subheader("Inspect an uploaded PCB image")
    mode = st.radio("Image inference method", ["Full image", "Overlapping tiles"], horizontal=True, key="upload_mode")
    if mode == "Overlapping tiles":
        t1, t2, t3 = st.columns(3)
        tile_w = t1.select_slider("Tile width", [320, 384, 480, 512, 640], value=480, key="upload_tile_w")
        tile_h = t2.select_slider("Tile height", [256, 320, 360, 384, 512], value=360, key="upload_tile_h")
        overlap = t3.slider("Tile overlap", 0.0, 0.40, 0.20, 0.05, key="upload_overlap")
        nms_iou = st.slider("Duplicate-box suppression (NMS IoU)", 0.20, 0.70, 0.45, 0.05, key="upload_nms")

    uploaded = st.file_uploader("Choose a PCB image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="pcb_upload")
    if uploaded is not None:
        try:
            image_rgb = np.array(Image.open(uploaded).convert("RGB"))
            st.image(image_rgb, caption=f"Uploaded image: {uploaded.name} ({image_rgb.shape[1]} × {image_rgb.shape[0]})", use_container_width=True)
            if st.button("▶ Run inspection", type="primary", key="upload_inspect"):
                weights_path = selected_weights_path
                if not weights_path.is_file():
                    st.error(f"Model weights not found: {weights_path}")
                else:
                    with st.spinner("Running model inference..."):
                        model = load_model(str(weights_path))
                        if mode == "Full image":
                            boxes, scores, classes, tile_count, raw_count = predict_full(model, image_rgb, confidence, imgsz)
                        else:
                            boxes, scores, classes, tile_count, raw_count = predict_tiles(
                                model, image_rgb, confidence, imgsz, tile_w, tile_h, overlap, nms_iou
                            )
                        names = model.names
                        annotated = draw_predictions(image_rgb, boxes, scores, classes, names)
                    left, right = st.columns(2)
                    with left:
                        st.image(image_rgb, caption="Original image", use_container_width=True)
                    with right:
                        st.image(annotated, caption="Detection result", use_container_width=True)
                    show_summary(boxes, scores, classes, names)
                    if mode == "Overlapping tiles":
                        st.caption(f"Tiles processed: {tile_count}. Raw detections before duplicate suppression: {raw_count}.")
                    st.warning("Predictions are not confirmed defects. Verify detections visually.")
                    buf = io.BytesIO()
                    Image.fromarray(annotated).save(buf, format="PNG")
                    st.download_button("⬇ Download annotated PCB", data=buf.getvalue(), file_name="pcb_inspection_result.png", mime="image/png")
        except Exception as exc:
            st.error(f"Could not open or inspect the uploaded image: {exc}")

with camera_tab:
    st.subheader("Live PCB inspection")
    st.markdown(
        "Choose a built-in laptop camera, any detected USB webcam, or connect DroidCam "
        "directly over Wi-Fi. The green rectangle is the region of interest (ROI), "
        "and YOLO receives only the pixels inside that rectangle."
    )

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "last_roi_result" not in st.session_state:
        st.session_state.last_roi_result = None
    if "camera_reader" not in st.session_state:
        st.session_state.camera_reader = None
    if "inference_worker" not in st.session_state:
        st.session_state.inference_worker = None
    if "last_inference_submit" not in st.session_state:
        st.session_state.last_inference_submit = 0.0

    source_type = st.selectbox(
        "Camera source",
        ["Laptop Camera", "USB Camera", "DroidCam"],
        key="camera_source_type",
    )

    selected_device = None
    stream_url = None
    if source_type in ("Laptop Camera", "USB Camera"):
        refresh_col, status_col = st.columns([1, 3])
        if refresh_col.button("↻ Refresh cameras", key="refresh_cameras"):
            discover_physical_cameras.clear()
            st.rerun()

        if enumerate_cameras is None:
            st.error(
                "Camera discovery package is missing. Install it with "
                r"`.venv\Scripts\python.exe -m pip install cv2-enumerate-cameras`."
            )
            physical_cameras = []
        else:
            with st.spinner("Detecting connected cameras..."):
                physical_cameras = discover_physical_cameras()

        category_cameras = [d for d in physical_cameras if d["category"] == source_type]
        if category_cameras:
            device_labels = [
                f'{device["name"]}  (VID: {device["vid"]}, PID: {device["pid"]})'
                for device in category_cameras
            ]
            selected_label = st.selectbox(
                "Detected device", device_labels, key=f"{source_type}_device_choice"
            )
            selected_device = category_cameras[device_labels.index(selected_label)]
            st.caption(
                f'{len(category_cameras)} {source_type.lower()} device(s) detected. '
                "Camera identifiers are discovered on this computer at runtime."
            )
        else:
            st.warning(
                f"No {source_type.lower()} was confidently identified. "
                "Connect the camera and use Refresh cameras. Some built-in webcams "
                "may be listed under USB Camera because Windows does not label every "
                "internal camera consistently."
            )
            if physical_cameras:
                with st.expander("Other detected cameras"):
                    for device in physical_cameras:
                        st.write(f'• {device["name"]} ({device["category"]})')

    else:
        droid_col1, droid_col2 = st.columns(2)
        droid_ip = droid_col1.text_input(
            "DroidCam IP address", value="10.119.97.61", key="droidcam_ip",
            placeholder="For example: 192.168.1.100",
        )
        droid_port = droid_col2.number_input(
            "DroidCam port", min_value=1, max_value=65535, value=4747,
            step=1, key="droidcam_port",
        )
        st.caption(
            "Keep DroidCam open on your phone. Your phone and computer must be able "
            "to communicate over the network. The IP address may change when you reconnect."
        )
        if droid_ip.strip():
            stream_url = f"http://{droid_ip.strip()}:{int(droid_port)}/video"

    start_col, stop_col = st.columns(2)
    can_start = (selected_device is not None) if source_type != "DroidCam" else bool(stream_url)

    if start_col.button(
        "▶ Start camera", type="primary", key="start_camera", disabled=not can_start
    ):
        old_reader = st.session_state.get("camera_reader")
        if old_reader is not None:
            old_reader.stop()
        old_worker = st.session_state.get("inference_worker")
        if old_worker is not None:
            old_worker.stop()

        if source_type == "DroidCam":
            cap = open_camera("DroidCam", stream_url=stream_url)
            camera_label = "DroidCam"
        else:
            capture = selected_device["captures"][0]
            cap = open_camera(
                source_type, camera_index=capture["index"], backend=capture["backend"]
            )
            camera_label = selected_device["name"]

        if cap is None:
            st.session_state.camera_running = False
            st.session_state.camera_reader = None
            if source_type == "DroidCam":
                st.error(
                    "Could not connect to DroidCam. Check the phone IP, port, "
                    "Wi-Fi connection, and that DroidCam is actively streaming."
                )
            else:
                st.error(
                    "Could not open the selected camera. Close other apps using it, "
                    "then refresh the camera list and try again."
                )
        else:
            st.session_state.camera_reader = CameraReader(cap)
            st.session_state.inference_worker = InferenceWorker()
            st.session_state.camera_running = True
            st.session_state.last_roi_result = None
            st.session_state.last_inference_submit = 0.0
            st.session_state.active_camera_label = camera_label

    if stop_col.button("■ Stop camera", key="stop_camera"):
        reader = st.session_state.get("camera_reader")
        if reader is not None:
            reader.stop()
        worker = st.session_state.get("inference_worker")
        if worker is not None:
            worker.stop()
        st.session_state.camera_reader = None
        st.session_state.inference_worker = None
        st.session_state.camera_running = False
        st.session_state.last_roi_result = None

    if st.session_state.camera_running:
        st.caption(
            f'Active source: {st.session_state.get("active_camera_label", "Camera")}. '
            "Adjust the ROI rectangle, then run continuous inference or inspect one frame."
        )
        r1, r2, r3, r4 = st.columns(4)
        roi_left = r1.slider("Box left (%)", 0, 90, 15, 1, key="roi_left")
        roi_top = r2.slider("Box top (%)", 0, 90, 15, 1, key="roi_top")
        roi_width = r3.slider("Box width (%)", 10, 100, 70, 1, key="roi_width")
        roi_height = r4.slider("Box height (%)", 10, 100, 70, 1, key="roi_height")

        @st.fragment(run_every=0.12)
        def live_camera_panel():
            reader = st.session_state.get("camera_reader")
            if reader is None:
                st.warning("No active camera. Stop and start the camera again.")
                return

            frame_bgr = reader.read_latest()
            if frame_bgr is None:
                st.info("Waiting for the first camera frame...")
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            x1 = int(w * roi_left / 100)
            y1 = int(h * roi_top / 100)
            x2 = min(w, x1 + max(1, int(w * roi_width / 100)))
            y2 = min(h, y1 + max(1, int(h * roi_height / 100)))
            preview = frame_rgb.copy()
            cv2.rectangle(
                preview, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)),
                (0, 255, 80), 3
            )
            cv2.putText(
                preview, "YOLO ROI", (x1 + 5, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2, cv2.LINE_AA
            )

            left, right = st.columns(2)
            with left:
                st.image(
                    preview, caption=f"Live camera feed with ROI box ({w} × {h})",
                    use_container_width=True
                )

            roi_rgb = frame_rgb[y1:y2, x1:x2].copy()
            live_inference = st.checkbox(
                "Run YOLO continuously on the boxed area",
                value=False, key="live_inference"
            )
            run_once = st.button(
                "🔎 Analyze current boxed area", key="analyze_roi_once"
            )

            if roi_rgb.size == 0:
                st.error("The selected ROI is empty. Adjust the box position or size.")
                return

            worker = st.session_state.get("inference_worker")
            should_submit = run_once
            if live_inference:
                now = time.monotonic()
                if now - st.session_state.last_inference_submit >= 0.5:
                    should_submit = True

            if should_submit:
                weights_path = selected_weights_path
                if not weights_path.is_file():
                    st.error(f"Model weights not found: {weights_path}")
                    return
                try:
                    model = load_model(str(weights_path))
                    if worker.submit(roi_rgb, model, confidence, imgsz):
                        st.session_state.last_inference_submit = time.monotonic()
                except Exception as exc:
                    st.error(f"Could not start live inference: {exc}")
                    return

            latest, inference_error, inference_busy = worker.snapshot()
            with right:
                if inference_error:
                    st.error(f"Live inference failed: {inference_error}")
                if latest is None:
                    st.image(
                        roi_rgb, caption="Exact image region sent to YOLO",
                        use_container_width=True
                    )
                    st.caption(
                        "Enable continuous inference or click Analyze current boxed area."
                    )
                else:
                    st.image(
                        latest["image"],
                        caption=(
                            f"YOLO result inside ROI only "
                            f"({latest['roi_size'][0]} × {latest['roi_size'][1]})"
                        ),
                        use_container_width=True
                    )
                    show_summary(
                        latest["boxes"], latest["scores"],
                        latest["classes"], latest["names"]
                    )
                if inference_busy:
                    st.caption("YOLO is processing the latest submitted ROI in the background.")

            st.caption(
                "Camera capture and preview run independently of YOLO inference. "
                "YOLO processes only the selected ROI, without tiling or segmentation."
            )

        live_camera_panel()
    else:
        st.info("Choose a camera source, select a detected device if needed, and press Start camera.")

st.divider()
st.caption(
    "Supported PCB model classes: mouse_bite, spur, missing_hole, short, open_circuit, spurious_copper. "
    "The DeepPCB model has a different class mapping. Confirm the selected model before interpreting labels."
)
