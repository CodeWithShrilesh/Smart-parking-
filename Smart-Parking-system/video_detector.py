import os
from datetime import datetime
from threading import Lock

import cv2
import numpy as np


def is_occupied(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )
    edges = cv2.Canny(blur, 70, 160)

    kernel = np.ones((3, 3), np.uint8)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    mask_ratio = cv2.countNonZero(adaptive) / (crop.shape[0] * crop.shape[1])
    edge_ratio = cv2.countNonZero(edges) / (crop.shape[0] * crop.shape[1])
    return mask_ratio > 0.14 or edge_ratio > 0.06


def format_time(dt_obj):
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds):
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class ParkingVideoTracker:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(base_path, "static", "output.jpg")
        self.default_image = os.path.join(base_path, "static", "parking.jpg")
        self.target_size = (1365, 1024)

        self.lock = Lock()
        self.source_path = self.default_image
        self.source_kind = "image"
        self.source_name = os.path.basename(self.source_path)
        self.cap = None

        self.slots = []
        self.detection_mode = "Auto"
        self.slot_states = {}
        self.entry_times = {}
        self.completed_sessions = []
        self.max_completed_sessions = 25
        self.frame_token = 0

        self.set_source(self.default_image)

    def _is_video(self, path):
        return os.path.splitext(path)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}

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

    def _read_initial_layout_frame(self):
        if self.source_kind == "video":
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return frame

        frame = cv2.imread(self.source_path)
        return frame

    def _read_frame(self):
        if self.source_kind == "video":
            if self.cap is None:
                return None

            ok, frame = self.cap.read()
            if ok:
                return frame

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.slot_states = {}
            self.entry_times = {}
            ok, frame = self.cap.read()
            return frame if ok else None

        return cv2.imread(self.source_path)

    def _fallback_grid_slots(self, width, height):
        slots = []
        rows = 5
        cols = 10
        margin_x = int(width * 0.08)
        margin_y = int(height * 0.12)
        usable_w = width - (2 * margin_x)
        usable_h = height - (2 * margin_y)

        slot_w = usable_w // cols
        slot_h = usable_h // rows

        for r in range(rows):
            for c in range(cols):
                x = margin_x + (c * slot_w)
                y = margin_y + (r * slot_h)
                slots.append((x, y, slot_w, slot_h))
        return slots

    def _auto_detect_slots(self, frame):
        img = cv2.resize(frame, self.target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 170)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = img.shape[0] * img.shape[1]
        min_area = int(frame_area * 0.0010)
        max_area = int(frame_area * 0.02)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            rect = cv2.minAreaRect(contour)
            (_, _), (w, h), _ = rect
            if w < 20 or h < 20:
                continue

            long_side = max(w, h)
            short_side = min(w, h)
            aspect_ratio = long_side / short_side
            if aspect_ratio < 1.6 or aspect_ratio > 5.2:
                continue

            box = cv2.boxPoints(rect).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(box)
            if bw < 25 or bh < 25:
                continue

            if y < int(img.shape[0] * 0.08):
                continue

            candidates.append((x, y, bw, bh))

        candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        deduped = []
        for cand in candidates:
            if all(iou(cand, existing) < 0.3 for existing in deduped):
                deduped.append(cand)

        deduped.sort(key=lambda b: (b[1], b[0]))
        if len(deduped) > 120:
            deduped = deduped[:120]

        if len(deduped) < 6:
            self.detection_mode = "Fallback Grid"
            return self._fallback_grid_slots(img.shape[1], img.shape[0])

        self.detection_mode = "Auto"
        return deduped

    def set_source(self, path):
        with self.lock:
            if not os.path.exists(path):
                return {"ok": False, "message": "File not found."}

            self.source_path = path
            self.source_kind = "video" if self._is_video(path) else "image"
            self.source_name = os.path.basename(path)
            self._open_source()

            layout_frame = self._read_initial_layout_frame()
            if layout_frame is None:
                return {"ok": False, "message": "Could not read uploaded media."}

            self.slots = self._auto_detect_slots(layout_frame)
            self.slot_states = {}
            self.entry_times = {}
            self.completed_sessions = []

            return {
                "ok": True,
                "slots_detected": len(self.slots),
                "source_kind": self.source_kind,
                "source_name": self.source_name
            }

    def process(self):
        with self.lock:
            frame = self._read_frame()
            self.frame_token += 1
            if frame is None:
                return {
                    "total": len(self.slots),
                    "free": 0,
                    "occupied": 0,
                    "active_sessions": [],
                    "recent_sessions": self.completed_sessions[:],
                    "source": "Unavailable",
                    "source_name": self.source_name,
                    "detection_mode": self.detection_mode,
                    "frame_token": self.frame_token
                }

            img = cv2.resize(frame, self.target_size)
            output = img.copy()
            now = datetime.now()
            free = 0

            for slot_index, (x, y, w, h) in enumerate(self.slots, start=1):
                crop = img[y:y + h, x:x + w]
                occupied = is_occupied(crop)
                was_occupied = self.slot_states.get(slot_index, False)

                if occupied and not was_occupied:
                    self.entry_times[slot_index] = now

                if not occupied and was_occupied:
                    in_time = self.entry_times.pop(slot_index, now)
                    duration = (now - in_time).total_seconds()
                    self.completed_sessions.insert(0, {
                        "slot": slot_index,
                        "in_time": format_time(in_time),
                        "out_time": format_time(now),
                        "duration": format_duration(duration),
                        "status": "Completed"
                    })
                    self.completed_sessions = self.completed_sessions[:self.max_completed_sessions]

                self.slot_states[slot_index] = occupied

                if occupied:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                    free += 1

                cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
                cv2.putText(
                    output,
                    f"S{slot_index}",
                    (x + 2, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )

            total = len(self.slots)
            occupied_count = total - free

            active_sessions = []
            for slot_index, in_time in self.entry_times.items():
                if self.slot_states.get(slot_index, False):
                    duration = (now - in_time).total_seconds()
                    active_sessions.append({
                        "slot": slot_index,
                        "in_time": format_time(in_time),
                        "out_time": "--",
                        "duration": format_duration(duration),
                        "status": "In Lot"
                    })

            active_sessions.sort(key=lambda item: item["slot"])
            cv2.imwrite(self.output_path, output)

            source_label = "Video" if self.source_kind == "video" else "Image"
            return {
                "total": total,
                "free": free,
                "occupied": occupied_count,
                "active_sessions": active_sessions,
                "recent_sessions": self.completed_sessions[:],
                "source": source_label,
                "source_name": self.source_name,
                "detection_mode": self.detection_mode,
                "frame_token": self.frame_token
            }


TRACKER = ParkingVideoTracker()


def set_media_source(path):
    return TRACKER.set_source(path)


def detect_parking():
    return TRACKER.process()
