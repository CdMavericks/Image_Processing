import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
from deepface.modules import verification as dst
import numpy as np
import cv2
import base64
import json
import os
import time
import requests
import mysql.connector
from datetime import datetime
import pytz
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

#

DATABASE_NAME = "face_database.json"
MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 0.60   

# Camera 2 API endpoint
CAMERA_2_API_URL = "http://192.168.100.10:8081/api/initiate_tag"

# Timezone
LOCAL_TIMEZONE = pytz.timezone("Asia/Kolkata")


# MySQL DB
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")



app = FastAPI(title="Camera 1 – Face Recognition Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# in-RAM face embedding database
face_database = {}




class ImagePayload(BaseModel):
    image_base64: str



def get_db_connection():
    try:
        return mysql.connector.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME, port=DB_PORT,
            ssl_disabled=False
        )
    except mysql.connector.Error as err:
        print(f"[DB_ERROR] Failed DB connection: {err}")
        return None


def get_current_course(student_class: str):
    """Return the current course based on timetable and IST time."""
    db = get_db_connection()
    if not db:
        return "DB Error"

    try:
        cursor = db.cursor()
        now = datetime.now(LOCAL_TIMEZONE)
        cur_time = now.strftime('%H:%M:%S')
        cur_day = now.strftime('%A')

        q = """
            SELECT course FROM timetable
            WHERE class=%s AND day_of_week=%s
            AND start_time <= %s AND end_time > %s
            LIMIT 1
        """

        cursor.execute(q, (student_class, cur_day, cur_time, cur_time))
        result = cursor.fetchone()
        db.close()

        if result:
            return result[0]
        return "General Session / No Class"

    except Exception as e:
        print(f"[TIMETABLE_ERROR] {e}")
        db.close()
        return "Unknown Course"


def log_attendance(usn: str):
    """Mark attendance once per class per day."""
    db = get_db_connection()
    if not db:
        return False

    try:
        cursor = db.cursor()

        # 1. Fetch student details
        cursor.execute("SELECT name, class FROM students WHERE usn=%s", (usn,))
        data = cursor.fetchone()
        if not data:
            print(f"[DB_LOG] Student {usn} not found.")
            db.close()
            return False

        name, student_class = data

        # 2. Determine current course
        current_course = get_current_course(student_class)
        if current_course in ["DB Error", "Unknown Course", "General Session / No Class"]:
            db.close()
            return False

        # 3. Check duplicate attendance
        today = datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d')
        q_check = """
            SELECT COUNT(*) FROM attendance
            WHERE usn=%s AND class=%s AND course=%s
            AND DATE(entry_time)=%s
        """
        cursor.execute(q_check, (usn, student_class, current_course, today))

        if cursor.fetchone()[0] > 0:
            print(f"[DB_LOG] {usn} already marked for {current_course}.")
            db.close()
            return False

        # 4. Insert attendance
        timestamp = datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        q_log = """
            INSERT INTO attendance (usn, class, attendance, course, entry_time)
            VALUES (%s, %s, %s, %s, %s)
        """

        cursor.execute(q_log, (usn, student_class, "P", current_course, timestamp))
        db.commit()
        cursor.close()
        db.close()

        print(f"[DB_LOG] Attendance recorded for {usn} in {current_course}.")
        return True

    except Exception as e:
        print(f"[ATTENDANCE_ERROR] {e}")
        db.close()
        return False


def decode_image(base64_string):
    try:
        if "," in base64_string:  
            base64_string = base64_string.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img

    except Exception as e:
        print(f"[DECODE_ERROR] {e}")
        return None


# STARTUP
@app.on_event("startup")
def load_embeddings():
    global face_database

    if not os.path.exists(DATABASE_NAME):
        print(f"[ERROR] Missing {DATABASE_NAME}. Run enrollment again.")
        return

    with open(DATABASE_NAME, "r") as f:
        face_database = json.load(f)

    print(f"[INFO] Loaded {len(face_database)} students from embedding DB.")

    try:
        print("[INFO] Loading ArcFace model...")
        DeepFace.build_model(MODEL_NAME)
        print("[INFO] ArcFace loaded successfully.")
    except Exception as e:
        print(f"[MODEL_ERROR] {e}")



@app.post("/api/identify")
async def identify_face(payload: ImagePayload):
    start = time.time()

    live_img = decode_image(payload.image_base64)
    if live_img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # DeepFace embedding extraction
    try:
        live_embedding_obj = DeepFace.represent(
            img_path=live_img,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
        live_emb = np.array(live_embedding_obj[0]["embedding"], dtype=np.float32)

    except ValueError:
        raise HTTPException(status_code=404, detail="No face detected.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {e}")

    # Compare with DB
    best_usn = None
    best_dist = float("inf")

    for usn, stored_list in face_database.items():
        for saved_emb in stored_list:
            try:
                d = dst.find_cosine_distance(live_emb, np.array(saved_emb, dtype=np.float32))
                if d < best_dist:
                    best_dist = d
                    best_usn = usn
            except:
                continue

    print(f"[MATCH] Best: {best_usn}  Dist: {best_dist:.4f}")

    # Match success
    if best_dist < DISTANCE_THRESHOLD:
        # Mark attendance for *current* class
        log_attendance(best_usn)

        # Notify Camera 2 (handoff)
        try:
            requests.post(
                CAMERA_2_API_URL,
                json={"usn": best_usn},   
                timeout=0.2
            )
            print(f"[HAND-OFF] Sent handoff for {best_usn} → Camera 2")

        except:
            print("[HAND-OFF_WARN] Could not reach Camera 2.")

        return {
            "status": "success",
            "usn": best_usn,
            "distance": float(best_dist)
        }

    else:
        raise HTTPException(status_code=404, detail="Face not recognized.")

# MAIN
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


