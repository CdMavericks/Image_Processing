import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import requests
import base64
import time
import json
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from datetime import datetime
import pytz
from dotenv import load_dotenv
import os

# --- CONFIG --- (Change based on the Wifi set up)
VIDEO_SOURCE ="http://192.168.100.10:8081/video"    
AUTH_API_URL = "http://192.168.100.5:8081/api/identify"
MODEL_NAME = "ArcFace"
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


# MySQL DB
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

LOCAL_TIME = pytz.timezone('Asia/Kolkata')
REVERIFY_COOLDOWN = 1200     # ~2 mins if running ~10FPS
ATTENDANCE_CHECK_INTERVAL = 30

# ---------- NEW: connection pool settings ----------
DB_POOL_NAME = "cam2_pool"
DB_POOL_SIZE = 5  # increase if you expect concurrency

# Shared state
lock = threading.Lock()
known_tags = {}           # tracker_id -> usn
cooldowns = {}            # tracker_id -> cooldown ticks
student_to_find = {}      # handoff dict from POST
last_frame = None
last_attendance_check = time.time()


last_attendance_logged = {}

# NEW: prevent repeated identification attempts on same tracker id while pending
pending_identifications = set()


IDENTIFY_SEMAPHORE = threading.BoundedSemaphore(12)  # allow more parallel body-ID attempts

# ---- FASTAPI ----
cam2_app = FastAPI(title="Camera 2 Tracker API")

class Handoff(BaseModel):
    usn: str
    live_embedding: list

@cam2_app.post("/api/initiate_tag")
async def initiate_tag(payload: Handoff):
    global student_to_find
    with lock:
        student_to_find = {"usn": payload.usn}
    print(f"[C2] Handoff received for {payload.usn}")
    return {"ok": True}

@cam2_app.get("/stream")
async def stream_video():
    def gen():
        global last_frame
        while True:
            if last_frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_frame + b"\r\n"
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

def start_api():
    uvicorn.run(cam2_app, host="0.0.0.0", port=7000, log_level="warning")

# ---- DB UTILS ----
# Create a pool at startup
db_pool = None
def init_db_pool():
    global db_pool
    try:
        db_pool = MySQLConnectionPool(
            pool_name=DB_POOL_NAME,
            pool_size=DB_POOL_SIZE,
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
            database=DB_NAME, port=DB_PORT,
        )
        print("[C2][DB] Connection pool created")
    except Exception as e:
        db_pool = None
        print("[C2][DB_ERR] Failed to create pool:", e)

def get_db_conn():
    global db_pool
    if db_pool:
        try:
            return db_pool.get_connection()
        except Exception as e:
            print("[C2][DB_WARN] Pool get_connection failed, falling back:", e)
    # Fallback to direct connect
    return mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, port=DB_PORT
    )

def log_attendance(usn):
    """
    Original behaviour preserved:
    - check students table for class
    - check timetable for current course/time
    - avoid duplicate in attendance table for same day
    Returns True if new attendance entry inserted, False otherwise.
    """
    try:
        # quick in-memory throttle to avoid hitting DB too frequently for same USN
        now_epoch = time.time()
        # If we've logged very recently (e.g., within 10 seconds) skip immediate re-log
        if last_attendance_logged.get(usn, 0) + 10 > now_epoch:
            return False

        conn = get_db_conn()
        cur = conn.cursor()

        cur.execute("SELECT class FROM students WHERE usn=%s", (usn,))
        r = cur.fetchone()
        if not r:
            cur.close()
            conn.close()
            return False

        student_class = r[0]
        now = datetime.now(LOCAL_TIME)
        today = now.strftime("%Y-%m-%d")

        cur.execute("""
        SELECT course FROM timetable
        WHERE class=%s AND day_of_week=%s AND start_time<=%s AND end_time>%s
        LIMIT 1
        """, (student_class, now.strftime("%A"), now.strftime("%H:%M:%S"), now.strftime("%H:%M:%S")))
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return False

        course = row[0]
        if course in ["General Session / No Class", "Unknown Course"]:
            cur.close()
            conn.close()
            return False

        # Extra check: in-memory per-day guard to avoid hitting DB for duplicates
        last_logged_date = last_attendance_logged.get(f"{usn}_date")
        if last_logged_date == today:
            cur.close()
            conn.close()
            return False

        cur.execute("""
        SELECT COUNT(*) FROM attendance
        WHERE usn=%s AND class=%s AND course=%s AND DATE(entry_time)=%s
        """, (usn, student_class, course, today))

        if cur.fetchone()[0] > 0:
            # mark memory cache to avoid further checks this day
            last_attendance_logged[f"{usn}_date"] = today
            cur.close()
            conn.close()
            return False

        cur.execute("""
        INSERT INTO attendance(usn, class, attendance, course, entry_time)
        VALUES(%s, %s, 'P', %s, %s)
        """, (usn, student_class, course, now.strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        cur.close()
        conn.close()

        # update memory cache
        last_attendance_logged[usn] = now_epoch
        last_attendance_logged[f"{usn}_date"] = today

        print(f"[C2][DB] Marked attendance for {usn} in {course}")
        return True

    except Exception as e:
        print("[C2][DB_ERR]", e)
        return False

# ---- IDENTIFICATION HELPERS ----
def call_identify_api(encoded_image_b64, timeout=2.0, extra=None):
    """
    Calls AUTH_API_URL with payload expected by your identify service.
    We send the body crop (base64) and include an optional 'mode' or 'extra' dict.
    Expected response: JSON with keys like {"success": True, "usn": "<USN>", ...}
    If your identify API accepts a 'mode' or different payload, adjust here.
    """
    try:
        payload = {"image": encoded_image_b64, "model": MODEL_NAME}
        if extra:
            payload.update(extra)
        resp = requests.post(AUTH_API_URL, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("success") and data.get("usn"):
            return data["usn"]
        return None
    except Exception as e:
        print("[C2][IDENT_API_ERR]", e)
        return None

def identify_body_async(tid, body_crop):
    """
    Runs in a separate thread. Tries multiple crops of the body (full, upper, lower)
    to increase chance of identification from any visible body part.
    Updates known_tags and cooldowns under lock if identification succeeds.
    """
    # Acquire semaphore to bound threads
    acquired = IDENTIFY_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        # Too many concurrent identifies - skip this attempt
        with lock:
            pending_identifications.discard(tid)
        return

    try:
        # Prepare multiple crops: full, upper-half, lower-half
        crops = []
        try:
            h, w = body_crop.shape[:2]
            if h <= 0 or w <= 0:
                return
            # Full person crop
            crops.append(body_crop)
            # Upper half
            crops.append(body_crop[0: max(1, h//2), :].copy())
            # Lower half
            crops.append(body_crop[max(0, h//2):h, :].copy())
            # Optional center crop
            center_h1 = max(0, h//4)
            center_h2 = min(h, center_h1 + h//2)
            crops.append(body_crop[center_h1:center_h2, :].copy())
        except Exception as e:
            print("[C2][IDENT_CROP_ERR]", e)
            return

        identified_usn = None
        # Try each crop sequentially (stop at first success)
        for idx, crop in enumerate(crops):
            try:
                _, buffer = cv2.imencode(".jpg", crop)
                encoded = base64.b64encode(buffer).decode()
            except Exception as e:
                print("[C2][IDENT_ENCODE_ERR]", e)
                continue

            # Send an extra flag so the identify service knows this is a body-based image
            usn = call_identify_api(encoded, timeout=2.0, extra={"mode": "body", "crop_index": idx})
            if usn:
                identified_usn = usn
                break

        if identified_usn:
            with lock:
                # only assign if still not known (avoid overwriting a handoff)
                if tid not in known_tags:
                    known_tags[tid] = identified_usn
                    cooldowns[tid] = REVERIFY_COOLDOWN
                    print(f"[C2][IDENTIFY] Body-identified {identified_usn} for ID {tid}")
    finally:
        with lock:
            pending_identifications.discard(tid)
        IDENTIFY_SEMAPHORE.release()


# ---- TRACKER ----
def run_tracker():
    global last_frame, student_to_find, known_tags, cooldowns, last_attendance_check

    # Initialize YOLO
    model = YOLO("yolov8n.pt")
    model.overrides['imgsz'] = 320   # PERFORMANCE BOOST

    tracker = sv.ByteTrack()

    GREEN = sv.Color.from_hex("#00FF00")
    WHITE = sv.Color.from_hex("#FFFFFF")

    box_annot = sv.BoxAnnotator(thickness=2)
    label_annot = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Multiple attempts to open stream, with backoff
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[C2] Cannot open camera stream, retrying...")
    retry_delay = 1.0
    while not cap.isOpened():
        time.sleep(retry_delay)
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        retry_delay = min(5.0, retry_delay * 1.5)
        if cap.isOpened():
            print("[C2] Reconnected to camera stream")

    print("[C2] Running on http://127.0.0.1:8081/stream")

    frame_count = 0
    last_detections = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # Try to reconnect
            print("[C2] Frame read failed — attempting reconnect")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                time.sleep(1.0)
            continue

        frame_count += 1

        # --- YOLO EVERY 3 FRAMES ---
        if frame_count % 3 == 0:
            results = model(frame, classes=[0], verbose=False, conf=0.5)[0]
            detections = sv.Detections.from_ultralytics(results)
            last_detections = detections
        else:
            detections = last_detections

        if detections is None:
            continue

        tracked = tracker.update_with_detections(detections)
        labels = []
        visible_usns = set()

        # --- UPDATE COOLDOWNS ---
        with lock:
            expired = [tid for tid, cd in list(cooldowns.items()) if cd <= 0]
            for tid in expired:
                known_tags.pop(tid, None)
                cooldowns.pop(tid, None)

            for tid in list(cooldowns.keys()):
                cooldowns[tid] -= 1

        # --- MAIN LOOP ---
        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            box_annot.color = WHITE
            label = "Unknown"

            with lock:
                # HANDOFF CHECK (original behaviour preserved)
                if student_to_find and tid not in known_tags:
                    usn = student_to_find.get("usn")
                    if usn:
                        known_tags[tid] = usn
                        cooldowns[tid] = REVERIFY_COOLDOWN
                        # clear handoff after using
                        student_to_find = {}
                        box_annot.color = GREEN
                        label = f"{usn}"
                        visible_usns.add(usn)
                        labels.append(label)
                        continue

                if tid in known_tags:
                    usn = known_tags[tid]
                    label = usn
                    visible_usns.add(usn)
                    cooldowns[tid] = REVERIFY_COOLDOWN
                    box_annot.color = GREEN

            # If still unknown, attempt auto-identify using body crops (non-blocking)
            if label == "Unknown":
                # Try to enqueue an identify if not already pending
                should_spawn = False
                with lock:
                    if tid not in pending_identifications and tid not in known_tags:
                        pending_identifications.add(tid)
                        should_spawn = True

                if should_spawn:
                    # crop box with padding and safe bounds (this is the body crop)
                    h, w = frame.shape[:2]
                    pad_x = int((x2 - x1) * 0.15)  # pad relative to box size
                    pad_y = int((y2 - y1) * 0.10)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w, x2 + pad_x)
                    cy2 = min(h, y2 + pad_y)
                    body_crop = frame[cy1:cy2, cx1:cx2].copy()

                    # spawn a thread to do identification asynchronously based on body appearance
                    thr = threading.Thread(target=identify_body_async, args=(tid, body_crop), daemon=True)
                    thr.start()

            labels.append(label)

        # --- ANNOTATE FAST ---
        annotated = frame.copy()
        annotated = box_annot.annotate(annotated, tracked)
        annotated = label_annot.annotate(annotated, tracked, labels)

        # FAST MJPEG ENCODE
        _, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        last_frame = jpeg.tobytes()

        # --- PERIODIC ATTENDANCE ---
        if time.time() - last_attendance_check >= ATTENDANCE_CHECK_INTERVAL:
            # copy set to avoid concurrency issues
            to_log = set(visible_usns)
            for usn in to_log:
                # use in-memory cooldown to reduce duplicate db writes
                try:
                    log_attendance(usn)
                except Exception as e:
                    print("[C2] [LOG_ATT_ERR]", e)
            last_attendance_check = time.time()


# ---- MAIN ----
if __name__ == "__main__":
    # initialize DB pool
    init_db_pool()

    threading.Thread(target=start_api, daemon=True).start()
    time.sleep(1)
    run_tracker()