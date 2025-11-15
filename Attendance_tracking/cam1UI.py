import streamlit as st
import cv2, base64, time, requests, numpy as np, keyboard

# --- CONFIG ---
BACKEND_API_URL = "http://127.0.0.1:8000/api/identify"
FACE_STABLE_TIME = 2           # seconds to hold face steady
COOLDOWN_SECONDS = 10           # skip same student for this many seconds
CAM2_TIMEOUT = 10            
# ----------------

st.set_page_config(layout="centered", page_title="ClassSight — Auto Attendance")
st.title("🎥 ClassSight — Smart Auto Attendance")
st.caption("Stand still for 2 seconds. System auto-captures and verifies you.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

run = st.checkbox("Start Camera")
frame_window = st.image([])
recently_marked = {}  # {usn: timestamp}


def verify_with_cam2_mock():
    """Simulate Camera2 verification (Alt+V to mock)"""
    st.info("⏳ Waiting for Camera 2 body verification...")
    start_wait = time.time()
    while True:
        if keyboard.is_pressed("alt+v"):
            st.success("✅ Camera 2 verification mocked!")
            return True
        if time.time() - start_wait > CAM2_TIMEOUT:
            st.warning("⚠️ No Cam2 response (timed out).")
            return False
        time.sleep(0.1)

if run:
    #IP_STREAM = "http://192.168.100.10:8081/?action=stream"  #change IP accordingly 

    #cap = cv2.VideoCapture(IP_STREAM)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera not found!")
    else:
        st.info("Camera active. Waiting for a face...")

    face_visible_since = None
    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Frame not available.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # Draw rectangle around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Stable face detection logic
        if len(faces) > 0:
            if face_visible_since is None:
                face_visible_since = time.time()
            elif time.time() - face_visible_since >= FACE_STABLE_TIME and not captured:
                captured = True
                st.info("📸 Face stable — capturing frame...")

                # Encode frame to base64
                _, buffer = cv2.imencode(".jpg", frame)
                img_base64 = base64.b64encode(buffer).decode("utf-8")

                try:
                    payload = {"image_base64": img_base64}
                    res = requests.post(BACKEND_API_URL, json=payload, timeout=8)

                    if res.status_code == 200:
                        data = res.json()
                        usn = data.get("usn", "Unknown")

                        # Check cooldown
                        now = time.time()
                        if usn in recently_marked and (now - recently_marked[usn] < COOLDOWN_SECONDS):
                            st.warning(f"⏳ {usn} already marked recently — skipping.")
                            captured = False
                            face_visible_since = None
                            continue

                        st.write(f"🧠 Recognized as {usn}")
                        
                        CAM2_HANDOFF_COOLDOWN = 20
                        
                        if usn in recently_marked and (time.time() - recently_marked[usn]) < CAM2_HANDOFF_COOLDOWN:
                            st.warning("Already handed off recently — skipping.")
                            continue
                        recently_marked[usn] = time.time()
                        st.success(f"✅ Attendance Logged! ({usn}) - C2 Tagger Initiated.")
                        # ---- SEND HANDOFF TO CAM 2 ----
                        try:
                            CAM2_URL = "http://192.168.100.5:7000/api/initiate_tag"   # change IP to your CAM2
                            handoff_payload = {
                            "usn": usn,
                            "live_embedding": []   # not used but required by your CAM2 API
                            }
                            hres = requests.post(CAM2_URL, json=handoff_payload, timeout=3)

                            if hres.status_code == 200:
                                st.success("📡 Handoff sent to Camera 2!")
                            else:
                                st.error(f"Cam2 error: {hres.status_code}")
                        except Exception as e:
                            st.error(f"Cam2 connection failed: {e}")


                        # Reset for next student
                        captured = False
                        face_visible_since = None
                        st.info("🔄 Ready for next student...")
                        continue

                    elif res.status_code == 404:
                        st.error("❌ Face not recognized. Please enroll first.")
                    else:
                        st.error(f"Server Error: {res.status_code}")
                        st.write(res.text)
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            face_visible_since = None
            captured = False

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    st.success("Camera stopped.")