import cv2
import numpy as np
import os

ULTRALYTICS_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".ultralytics"
)
os.environ.setdefault("YOLO_CONFIG_DIR", ULTRALYTICS_CONFIG_DIR)
os.makedirs(ULTRALYTICS_CONFIG_DIR, exist_ok=True)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def build_slots(row_x, row_y, row_w, row_h, num_slots):
    slots = []
    slot_w = row_w / num_slots
    for i in range(num_slots):
        x1 = int(row_x + i * slot_w)
        x2 = int(row_x + (i + 1) * slot_w)
        slots.append((x1, row_y, x2 - x1, row_h))
    return slots

def is_occupied(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, dark = cv2.threshold(blur, 112, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    count = cv2.countNonZero(dark)
    area = crop.shape[0] * crop.shape[1]
    ratio = count / area

    return ratio > 0.102

def detect_parking():
    base_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_path, "static", "parking.jpg")

    img = cv2.imread(image_path)
    if img is None:
        print("❌ ERROR: Image NOT loading")
        return 0, 0, 0

    img = cv2.resize(img, (1365, 1024))
    output = img.copy()

    slots = []

    # TOP ROW
    top_row = build_slots(
        row_x=148,
        row_y=188,
        row_w=1214,
        row_h=144,
        num_slots=17
    )
    slots.extend(top_row)

    # MIDDLE ROW
    middle_row = build_slots(
        row_x=202,
        row_y=512,
        row_w=1163,
        row_h=140,
        num_slots=16
    )
    slots.extend(middle_row)

    # BOTTOM ROW
    bottom_row = build_slots(
        row_x=228,
        row_y=705,
        row_w=1137,
        row_h=145,
        num_slots=15
    )
    slots.extend(bottom_row)

    free = 0

    for (x, y, w, h) in slots:
        crop = img[y:y+h, x:x+w]
        occupied = is_occupied(crop)

        if occupied:
            color = (0, 0, 255)   # Red
        else:
            color = (0, 255, 0)   # Green
            free += 1

        # Thin clean boxes
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)

    total = len(slots)
    occupied = total - free

    output_path = os.path.join(base_path, "static", "output.jpg")
    cv2.imwrite(output_path, output)

    return total, free, occupied

if __name__ == "__main__":
    print(detect_parking())

# -----------------------------
# Refactored Smart Detector API
# -----------------------------
from datetime import datetime
from threading import Lock

from database import (
    clear_active_records,
    end_parking,
    get_current_parked,
    get_history,
    init_db,
    start_parking,
)


def _format_time(dt_obj):
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds):
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _duration_to_seconds(duration_text):
    if not duration_text or duration_text == "--":
        return 0
    parts = duration_text.split(":")
    if len(parts) != 3:
        return 0
    hours, minutes, seconds = [int(part) for part in parts]
    return (hours * 3600) + (minutes * 60) + seconds


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _intersection_over_slot(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    slot_area = aw * ah
    if slot_area <= 0:
        return 0.0
    return inter / slot_area


def _is_occupied_v2(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    edges = cv2.Canny(blur, 70, 170)

    kernel = np.ones((3, 3), np.uint8)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    area = crop.shape[0] * crop.shape[1]
    mask_ratio = cv2.countNonZero(adaptive) / area
    edge_ratio = cv2.countNonZero(edges) / area
    return mask_ratio > 0.14 or edge_ratio > 0.06


def _infer_vehicle_box_from_slot(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    edges = cv2.Canny(blur, 70, 170)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.bitwise_or(mask, edges)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = crop.shape[0] * crop.shape[1]
    min_area = crop_area * 0.12
    max_area = crop_area * 0.95
    best_box = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < crop.shape[1] * 0.25 or h < crop.shape[0] * 0.35:
            continue

        rect_area = w * h
        fill_ratio = area / max(1, rect_area)
        if fill_ratio < 0.28:
            continue

        if area > best_area:
            best_area = area
            best_box = (x, y, w, h)

    return best_box


class SmartParkingDetector:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.base_path = base_path
        self.output_path = os.path.join(base_path, "static", "output.jpg")
        self.default_image = os.path.join(base_path, "static", "parking.jpg")
        self.target_size = (1365, 1024)
        self.video_exts = {".mp4", ".avi"}

        self.lock = Lock()
        self.source_path = self.default_image
        self.source_kind = "image"
        self.source_name = os.path.basename(self.source_path)
        self.cap = None

        self.slots = []
        self.detection_mode = "Auto"
        self.yolo_status = "YOLO pending"
        self.yolo_model = None
        self.yolo_error = ""
        self.yolo_model_names = [
            "best.pt",
            "car_detector.pt",
            "yolov8m.pt",
            "yolov8s.pt",
            "yolov8n.pt",
        ]
        self.yolo_model_path = self._resolve_yolo_model_path()
        self.vehicle_class_ids = {2, 3, 5, 7}
        self.vehicle_confidence = 0.14
        self.video_vehicle_confidence = 0.1
        self.slot_overlap_threshold = 0.2
        self.frame_token = 0
        self.track_counter = 1
        self.tracks = {}
        self.track_history = []
        self.max_history_rows = 50
        self.max_track_misses = 8

        init_db()
        self._load_yolo_model()
        self.set_source(self.default_image)

    def _is_video(self, path):
        return os.path.splitext(path)[1].lower() in self.video_exts

    def _release_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _open_source(self):
        self._release_capture()
        if self.source_kind == "video":
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                self.cap = None

    def _read_layout_frame(self):
        if self.source_kind == "video":
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return frame
        return cv2.imread(self.source_path)

    def _read_frame(self):
        if self.source_kind == "video":
            if self.cap is None:
                return None

            ok, frame = self.cap.read()
            if ok:
                return frame

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.tracks = {}
            ok, frame = self.cap.read()
            return frame if ok else None

        return cv2.imread(self.source_path)

    def _resolve_yolo_model_path(self):
        configured_path = os.environ.get("YOLO_MODEL_PATH", "").strip()
        candidate_paths = []
        if configured_path:
            candidate_paths.append(configured_path)

        for model_name in self.yolo_model_names:
            candidate_paths.extend(
                [
                    os.path.join(self.base_path, "models", model_name),
                    os.path.join(self.base_path, model_name),
                ]
            )

        for path in candidate_paths:
            if path and os.path.exists(path):
                return path
        return None

    def _load_yolo_model(self):
        if YOLO is None:
            self.yolo_status = "YOLO unavailable"
            self.yolo_error = "ultralytics import failed"
            return

        self.yolo_model_path = self._resolve_yolo_model_path()
        if not self.yolo_model_path:
            self.yolo_status = "YOLO unavailable (missing local weights)"
            self.yolo_error = "no local model file found"
            return

        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_status = f"YOLO active ({os.path.basename(self.yolo_model_path)})"
            self.yolo_error = ""
        except Exception as exc:
            self.yolo_model = None
            self.yolo_status = "YOLO unavailable"
            self.yolo_error = str(exc)

    def _ensure_yolo_model(self):
        if self.yolo_model is not None:
            return

        self.yolo_model_path = self._resolve_yolo_model_path()
        if self.yolo_model_path:
            self._load_yolo_model()

    def _detect_vehicle_boxes(self, img):
        self._ensure_yolo_model()
        if self.yolo_model is None:
            return []

        vehicle_boxes = []
        inference_regions = self._build_inference_regions(img.shape[1], img.shape[0])
        confidence = (
            self.video_vehicle_confidence if self.source_kind == "video" else self.vehicle_confidence
        )

        try:
            for x1, y1, x2, y2 in inference_regions:
                crop = img[y1:y2, x1:x2]
                results = self.yolo_model.predict(
                    source=crop,
                    verbose=False,
                    conf=confidence,
                    classes=sorted(self.vehicle_class_ids),
                    imgsz=960,
                )
                vehicle_boxes.extend(self._extract_vehicle_boxes(results, x1, y1))
            self.yolo_error = ""
        except Exception as exc:
            self.yolo_model = None
            self.yolo_status = "YOLO unavailable"
            self.yolo_error = str(exc)
            return []

        filtered_boxes = self._filter_vehicle_boxes(vehicle_boxes, img.shape[1], img.shape[0])
        return self._dedupe_slots(filtered_boxes, overlap=0.4)

    def _build_inference_regions(self, width, height):
        regions = [(0, 0, width, height)]

        tile_cols = 3
        tile_rows = 3
        overlap_x = int(width * 0.1)
        overlap_y = int(height * 0.1)
        tile_w = width // tile_cols
        tile_h = height // tile_rows

        for row in range(tile_rows):
            for col in range(tile_cols):
                x1 = max(0, (col * tile_w) - overlap_x)
                y1 = max(0, (row * tile_h) - overlap_y)
                x2 = min(width, ((col + 1) * tile_w) + overlap_x)
                y2 = min(height, ((row + 1) * tile_h) + overlap_y)
                regions.append((x1, y1, x2, y2))

        return regions

    def _extract_vehicle_boxes(self, results, offset_x=0, offset_y=0):
        vehicle_boxes = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            cls_list = boxes.cls.cpu().numpy().astype(int)
            conf_list = boxes.conf.cpu().numpy()

            for xyxy, cls_id, confidence in zip(xyxy_list, cls_list, conf_list):
                if cls_id not in self.vehicle_class_ids:
                    continue
                threshold = (
                    self.video_vehicle_confidence
                    if self.source_kind == "video"
                    else self.vehicle_confidence
                )
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = xyxy.tolist()
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                vehicle_boxes.append((x1 + offset_x, y1 + offset_y, width, height))

        return vehicle_boxes

    def _filter_vehicle_boxes(self, boxes, frame_w, frame_h):
        filtered = []
        min_area = frame_w * frame_h * 0.0012
        max_area = frame_w * frame_h * 0.08

        for x, y, w, h in boxes:
            area = w * h
            if area < min_area or area > max_area:
                continue

            aspect_ratio = max(w, h) / max(1, min(w, h))
            if aspect_ratio > 4.2:
                continue

            if w < 28 or h < 28:
                continue

            # Tiny detections near lane markings and parking blocks are a common false positive.
            if w < 42 and h < 42:
                continue

            filtered.append((x, y, w, h))

        return filtered

    def _count_cars_in_frame(self, frame):
        img = cv2.resize(frame, self.target_size)
        vehicle_boxes = self._dedupe_slots(self._detect_vehicle_boxes(img), overlap=0.5)
        return len(vehicle_boxes)

    def _count_cars_in_video_preview(self):
        if self.cap is None:
            return 0, 0

        sample_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if sample_total <= 0:
            return 0, 0

        sample_points = sorted(
            {
                0,
                min(sample_total - 1, sample_total // 4),
                min(sample_total - 1, sample_total // 2),
                min(sample_total - 1, (3 * sample_total) // 4),
            }
        )
        best_count = 0
        best_frame_idx = 0
        original_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        for frame_idx in sample_points:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
            if not ok:
                continue
            count = self._count_cars_in_frame(frame)
            if count >= best_count:
                best_count = count
                best_frame_idx = frame_idx

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
        if best_count == 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        return best_count, best_frame_idx

    def _box_center(self, box):
        x, y, w, h = box
        return x + (w / 2), y + (h / 2)

    def _distance(self, box_a, box_b):
        ax, ay = self._box_center(box_a)
        bx, by = self._box_center(box_b)
        return float(np.hypot(ax - bx, ay - by))

    def _build_track_id(self):
        track_id = f"CAR-{self.track_counter:03d}"
        self.track_counter += 1
        return track_id

    def _update_tracks(self, detections, now):
        unmatched_tracks = set(self.tracks.keys())
        matched_detection_indexes = set()

        for det_idx, detection in enumerate(detections):
            best_track_id = None
            best_score = -1.0

            for track_id in list(unmatched_tracks):
                track = self.tracks[track_id]
                iou_score = _iou(detection, track["box"])
                distance = self._distance(detection, track["box"])
                if iou_score < 0.12 and distance > 120:
                    continue

                score = iou_score - (distance / 1000.0)
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is None:
                continue

            track = self.tracks[best_track_id]
            track["box"] = detection
            track["last_seen"] = now
            track["misses"] = 0
            unmatched_tracks.remove(best_track_id)
            matched_detection_indexes.add(det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detection_indexes:
                continue

            track_id = self._build_track_id()
            self.tracks[track_id] = {
                "box": detection,
                "first_seen": now,
                "last_seen": now,
                "misses": 0,
            }

        finished_tracks = []
        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track["misses"] += 1
            if track["misses"] > self.max_track_misses:
                finished_tracks.append((track_id, track))
                del self.tracks[track_id]

        for track_id, track in finished_tracks:
            duration_seconds = (track["last_seen"] - track["first_seen"]).total_seconds()
            self.track_history.insert(
                0,
                {
                    "vehicle_id": track_id,
                    "slot": "YOLO",
                    "entry_time": _format_time(track["first_seen"]),
                    "exit_time": _format_time(track["last_seen"]),
                    "duration_seconds": int(duration_seconds),
                    "duration": _format_duration(duration_seconds),
                },
            )
        self.track_history = self.track_history[: self.max_history_rows]

    def _build_snapshot_rows(self, detections):
        rows = []
        for idx, (vx, vy, vw, vh) in enumerate(detections, start=1):
            rows.append(
                {
                    "vehicle_id": f"CAR-{idx:03d}",
                    "slot": "YOLO",
                    "entry_time": "Snapshot",
                    "exit_time": "--",
                    "duration": "--",
                }
            )
        return rows

    def _build_active_track_rows(self, now):
        rows = []
        for track_id, track in sorted(self.tracks.items(), key=lambda item: item[0]):
            duration_seconds = (now - track["first_seen"]).total_seconds()
            x, y, w, h = track["box"]
            rows.append(
                {
                    "vehicle_id": track_id,
                    "slot": "YOLO",
                    "entry_time": _format_time(track["first_seen"]),
                    "exit_time": "--",
                    "duration": _format_duration(duration_seconds),
                }
            )
        return rows

    def _build_parking_summary(self, current_rows, history_rows):
        active_seconds = 0
        for row in current_rows:
            if row["entry_time"] == "Snapshot":
                continue
            duration_text = row["duration"].split(" | ")[0]
            active_seconds += _duration_to_seconds(duration_text)

        completed_seconds = 0
        for row in history_rows:
            completed_seconds += row.get("duration_seconds", _duration_to_seconds(row["duration"]))

        return {
            "active_time": _format_duration(active_seconds),
            "completed_time": _format_duration(completed_seconds),
            "sessions": len(history_rows),
        }

    def _dedupe_slots(self, boxes, overlap=0.35):
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep = []
        for box in boxes:
            if all(_iou(box, existing) < overlap for existing in keep):
                keep.append(box)
        keep.sort(key=lambda b: (b[1], b[0]))
        return keep

    def _fallback_grid_slots(self, width, height):
        slots = []
        rows = 5
        cols = 10
        margin_x = int(width * 0.08)
        margin_y = int(height * 0.12)
        usable_w = width - (2 * margin_x)
        usable_h = height - (2 * margin_y)
        slot_w = max(30, usable_w // cols)
        slot_h = max(30, usable_h // rows)

        for r in range(rows):
            for c in range(cols):
                x = margin_x + (c * slot_w)
                y = margin_y + (r * slot_h)
                slots.append((x, y, slot_w, slot_h))
        return slots

    def _slots_from_contours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = img.shape[0] * img.shape[1]
        min_area = int(frame_area * 0.0007)
        max_area = int(frame_area * 0.03)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            rect = cv2.minAreaRect(contour)
            (_, _), (w, h), _ = rect
            if w < 18 or h < 18:
                continue

            long_side = max(w, h)
            short_side = min(w, h)
            ratio = long_side / short_side
            if ratio < 1.3 or ratio > 6.0:
                continue

            box = cv2.boxPoints(rect).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(box)
            if bw < 25 or bh < 20:
                continue
            if y < int(img.shape[0] * 0.05):
                continue

            boxes.append((x, y, bw, bh))

        return self._dedupe_slots(boxes)

    def _cluster_y_levels(self, values, gap=90):
        if not values:
            return []
        values = sorted(values)
        groups = [[values[0]]]
        for v in values[1:]:
            if abs(v - groups[-1][-1]) <= gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [int(np.median(group)) for group in groups if len(group) >= 3]

    def _group_slots_by_row(self, boxes, gap=90):
        if not boxes:
            return []

        rows = []
        for box in sorted(boxes, key=lambda b: (b[1], b[0])):
            x, y, w, h = box
            center_y = y + (h / 2)
            placed = False
            for row in rows:
                row_center = np.median([item[1] + (item[3] / 2) for item in row])
                if abs(center_y - row_center) <= gap:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        normalized_rows = []
        for row in rows:
            row.sort(key=lambda b: b[0])
            if len(row) >= 3:
                normalized_rows.append(row)
        return normalized_rows

    def _normalize_row_slots(self, row):
        if len(row) < 2:
            return row

        widths = [box[2] for box in row]
        heights = [box[3] for box in row]
        ys = [box[1] for box in row]
        median_width = int(np.median(widths))
        median_height = int(np.median(heights))
        median_y = int(np.median(ys))

        if median_width <= 0:
            return row

        refined = []
        for x, y, w, h in row:
            estimated_slots = max(1, int(round(w / median_width)))
            if w > int(median_width * 1.55):
                estimated_slots = max(2, estimated_slots)

            if estimated_slots == 1:
                refined.append((x, median_y, w, median_height))
                continue

            split_width = max(24, int(round(w / estimated_slots)))
            for idx in range(estimated_slots):
                slot_x = x + (idx * split_width)
                if idx == estimated_slots - 1:
                    actual_width = (x + w) - slot_x
                else:
                    actual_width = split_width
                if actual_width >= 24:
                    refined.append((slot_x, median_y, actual_width, median_height))

        refined.sort(key=lambda b: b[0])
        return refined

    def _refine_slots(self, boxes):
        rows = self._group_slots_by_row(boxes, gap=95)
        refined = []
        for row in rows:
            refined.extend(self._normalize_row_slots(row))

        if not refined:
            return boxes

        return self._dedupe_slots(refined, overlap=0.45)

    def _slots_from_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 180)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80, minLineLength=35, maxLineGap=18
        )
        if lines is None:
            return []

        vertical_lines = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            length = np.hypot(dx, dy)
            if 60 <= angle <= 120 and length >= 35:
                vertical_lines.append((x1, y1, x2, y2, length))

        if len(vertical_lines) < 6:
            return []

        y_mids = [int((ln[1] + ln[3]) / 2) for ln in vertical_lines]
        row_centers = self._cluster_y_levels(y_mids, gap=110)
        boxes = []
        for row_center in row_centers:
            row_lines = [
                ln
                for ln in vertical_lines
                if abs(((ln[1] + ln[3]) / 2) - row_center) <= 70
            ]
            if len(row_lines) < 4:
                continue

            xs = sorted([int((ln[0] + ln[2]) / 2) for ln in row_lines])
            filtered_x = []
            for x in xs:
                if not filtered_x or abs(x - filtered_x[-1]) > 22:
                    filtered_x.append(x)
            if len(filtered_x) < 3:
                continue

            lengths = [ln[4] for ln in row_lines]
            slot_h = int(np.clip(np.median(lengths) * 0.95, 45, 240))
            y = int(np.clip(row_center - (slot_h / 2), 0, img.shape[0] - slot_h))

            for idx in range(len(filtered_x) - 1):
                x1 = filtered_x[idx]
                x2 = filtered_x[idx + 1]
                width = x2 - x1
                if 24 <= width <= 210:
                    boxes.append((x1, y, width, slot_h))

        return self._dedupe_slots(boxes, overlap=0.4)

    def _auto_detect_slots(self, frame):
        img = cv2.resize(frame, self.target_size)
        contour_slots = self._slots_from_contours(img)
        line_slots = self._slots_from_lines(img)
        refined_line_slots = self._refine_slots(line_slots)
        refined_contour_slots = self._refine_slots(contour_slots)

        if len(refined_line_slots) >= 8 and len(refined_line_slots) >= len(refined_contour_slots):
            self.detection_mode = "Auto (Line-Based Refined)"
            selected = refined_line_slots
        elif len(refined_contour_slots) >= 8:
            self.detection_mode = "Auto (Contour-Based Refined)"
            selected = refined_contour_slots
        else:
            merged = self._refine_slots(self._dedupe_slots(refined_contour_slots + refined_line_slots))
            if len(merged) >= 8:
                self.detection_mode = "Auto (Hybrid Refined)"
                selected = merged
            else:
                self.detection_mode = "Fallback Grid"
                selected = self._fallback_grid_slots(img.shape[1], img.shape[0])

        return selected[:160]

    def _build_vehicle_id(self):
        vehicle_id = f"V{self.vehicle_counter:05d}"
        self.vehicle_counter += 1
        return vehicle_id

    def set_source(self, path):
        with self.lock:
            self._ensure_yolo_model()
            if not os.path.exists(path):
                return {"ok": False, "message": "File not found."}

            self.source_path = path
            self.source_kind = "video" if self._is_video(path) else "image"
            self.source_name = os.path.basename(path)
            self._open_source()

            layout_frame = self._read_layout_frame()
            if layout_frame is None:
                return {"ok": False, "message": "Could not read uploaded media."}

            self.tracks = {}
            self.track_history = []
            self.track_counter = 1
            clear_active_records()
            if self.source_kind == "video":
                preview_count, _ = self._count_cars_in_video_preview()
            else:
                preview_count = self._count_cars_in_frame(layout_frame)
            return {
                "ok": True,
                "cars_detected": preview_count,
                "source_kind": self.source_kind,
                "source_name": self.source_name,
                "mode": self._build_detection_mode(),
            }

    def process(self):
        with self.lock:
            self._ensure_yolo_model()
            frame = self._read_frame()
            self.frame_token += 1
            if frame is None:
                return {
                    "total": 0,
                    "free": 0,
                    "occupied": 0,
                    "source": "Unavailable",
                    "source_name": self.source_name,
                    "detection_mode": self._build_detection_mode(),
                    "current_vehicles": [],
                    "history": [],
                    "parking_summary": {
                        "active_time": "00:00:00",
                        "completed_time": "00:00:00",
                        "sessions": 0,
                    },
                    "frame_token": self.frame_token,
                }

            img = cv2.resize(frame, self.target_size)
            output = img.copy()
            now = datetime.now()
            vehicle_boxes = self._dedupe_slots(self._detect_vehicle_boxes(img), overlap=0.5)
            if self.source_kind == "video":
                self._update_tracks(vehicle_boxes, now)

            for vx, vy, vw, vh in vehicle_boxes:
                cv2.rectangle(output, (vx, vy), (vx + vw, vy + vh), (255, 191, 0), 2)
                cv2.putText(
                    output,
                    "YOLO Car",
                    (vx, max(18, vy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 191, 0),
                    1,
                    cv2.LINE_AA,
                )

            total = len(vehicle_boxes)
            cv2.imwrite(self.output_path, output)

            if self.source_kind == "video":
                current_vehicles = self._build_active_track_rows(now)
                history = self.track_history[:]
            else:
                current_vehicles = self._build_snapshot_rows(vehicle_boxes)
                history = []
            parking_summary = self._build_parking_summary(current_vehicles, history)

            source_label = "Video" if self.source_kind == "video" else "Image"
            return {
                "total": total,
                "free": total,
                "occupied": 0,
                "source": source_label,
                "source_name": self.source_name,
                "detection_mode": self._build_detection_mode(),
                "current_vehicles": current_vehicles,
                "history": history,
                "parking_summary": parking_summary,
                "frame_token": self.frame_token,
            }

    def _build_detection_mode(self):
        if self.yolo_error:
            return f"YOLO Vehicle Detection | {self.yolo_status} | {self.yolo_error[:120]}"
        return f"YOLO Vehicle Detection | {self.yolo_status}"


_DETECTOR = SmartParkingDetector()


def set_media_source(path):
    return _DETECTOR.set_source(path)


def detect_parking():
    return _DETECTOR.process()
