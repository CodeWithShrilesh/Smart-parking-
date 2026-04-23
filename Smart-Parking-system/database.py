import csv
import os
from threading import Lock


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "static", "parking_logs.csv")
CSV_FIELDS = [
    "id",
    "vehicle_id",
    "plate_number",
    "slot_number",
    "entry_time",
    "exit_time",
    "duration_seconds",
    "status",
]
_LOCK = Lock()


def _ensure_csv():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def _read_all_rows():
    _ensure_csv()
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all_rows(rows):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def init_db():
    _ensure_csv()


def start_parking(vehicle_id, slot_number, entry_time, plate_number="N/A"):
    with _LOCK:
        rows = _read_all_rows()
        next_id = len(rows) + 1
        rows.append(
            {
                "id": str(next_id),
                "vehicle_id": vehicle_id,
                "plate_number": plate_number,
                "slot_number": str(slot_number),
                "entry_time": entry_time,
                "exit_time": "",
                "duration_seconds": "",
                "status": "IN",
            }
        )
        _write_all_rows(rows)


def end_parking(vehicle_id, slot_number, exit_time, duration_seconds):
    with _LOCK:
        rows = _read_all_rows()
        for row in reversed(rows):
            if (
                row["vehicle_id"] == vehicle_id
                and row["slot_number"] == str(slot_number)
                and row["status"] == "IN"
                and not row["exit_time"]
            ):
                row["exit_time"] = exit_time
                row["duration_seconds"] = str(int(duration_seconds))
                row["status"] = "OUT"
                break
        _write_all_rows(rows)


def clear_active_records():
    with _LOCK:
        rows = _read_all_rows()
        rows = [row for row in rows if row["status"] != "IN"]
        _write_all_rows(rows)


def _format_duration(duration_seconds):
    if duration_seconds in (None, "", "None"):
        return "--"
    total_seconds = max(0, int(duration_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_current_parked():
    with _LOCK:
        rows = _read_all_rows()
    rows = [row for row in rows if row["status"] == "IN"]
    rows.sort(key=lambda row: row["entry_time"], reverse=True)
    data = []
    for row in rows:
        data.append(
            {
                "vehicle_id": row["vehicle_id"],
                "plate_number": row["plate_number"] or "N/A",
                "slot": int(row["slot_number"]),
                "entry_time": row["entry_time"],
                "exit_time": "--",
                "duration": "--",
                "status": "In Lot",
            }
        )
    return data


def get_history(limit=100):
    with _LOCK:
        rows = _read_all_rows()
    rows = [row for row in rows if row["status"] == "OUT"]
    rows.sort(key=lambda row: int(row["id"]), reverse=True)
    rows = rows[:limit]
    data = []
    for row in rows:
        data.append(
            {
                "vehicle_id": row["vehicle_id"],
                "plate_number": row["plate_number"] or "N/A",
                "slot": int(row["slot_number"]),
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"] or "--",
                "duration": _format_duration(row["duration_seconds"]),
                "status": "Completed",
            }
        )
    return data
