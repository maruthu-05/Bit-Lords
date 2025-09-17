import cv2
import subprocess
import dlib
from imutils import face_utils
import numpy as np
import time
from ultralytics import YOLO

# Initialize constants
EYE_AR_THRESH = 0.25
EYE_CLOSED_TIME_THRESH = 4
MOUTH_AR_THRESH = 0.7

COUNTER = 0
YAWN_COUNTER = 0
start_time = None
yawn_start_time = None
score = 100
penalty_eye_closure = 0.22
penalty_yawning = 0.11

last_score_update = time.time()
score_update_interval = 1200

eye_alert_spoken = False
yawn_alert_spoken = False
yawn_in_progress = False

# Initialize YOLOv8 face detection model (replace with your face detection weights)
face_model = YOLO("yolov8n-face.pt")

# Initialize dlib shape predictor for landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize video capture
vs = cv2.VideoCapture(0)

def eye_aspect_ratio(eye):
    # EAR calculation code (implement)
    # Example:
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # MAR calculation code (implement)
    A = np.linalg.norm(mouth[13] - mouth[19])  # 14-20 indexing in 0-based
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    D = np.linalg.norm(mouth[12] - mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar

def speak_alert(message):
    print("ALERT:", message)

def get_voice_command():
    # Placeholder for voice recognition; return command string or None
    return None

def update_user_field(email, field, value):
    print(f"Updating {field} for {email} to {value}")

while True:
    ret, frame = vs.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    # Use YOLOv8 for face detection
    results = face_model(frame)
    faces = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for box in faces:
        x1, y1, x2, y2 = map(int, box)
        # Expand the box slightly if needed to include full face
        face_rect = dlib.rectangle(x1, y1, x2, y2)

        # Detect landmarks on the face ROI using dlib predictor on gray image
        shape = predictor(gray, face_rect)
        shape = face_utils.shape_to_np(shape)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Eye closure logic
        if ear < EYE_AR_THRESH:
            if start_time is None:
                start_time = time.time()
            COUNTER += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= EYE_CLOSED_TIME_THRESH and not eye_alert_spoken:
                alert_message = "Alert! Eyes closed for more than 4 seconds."
                cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                speak_alert(alert_message)
                eye_alert_spoken = True
                score = max(score - penalty_eye_closure, 0)

                alert_message_2 = "Are you feeling drowsy? Do you want to listen to some music?"
                speak_alert(alert_message_2)
                command = get_voice_command()
                if command:
                    if "yes" in command.lower():
                        subprocess.call('Spotify.exe')
                    elif "no" in command.lower():
                        print("User declined to listen to music.")
                    else:
                        print("Unrecognized command.")
                yawn_alert_spoken = True
                YAWN_COUNTER = 0
        else:
            COUNTER = 0
            start_time = None
            eye_alert_spoken = False
            cv2.putText(frame, "EYES OPEN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Yawning logic
        if mar > MOUTH_AR_THRESH:
            if not yawn_in_progress:
                yawn_start_time = time.time()
                yawn_in_progress = True
            else:
                yawn_duration = time.time() - yawn_start_time
                if yawn_duration >= 2:
                    cv2.putText(frame, "YAWNING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not yawn_alert_spoken:
                        YAWN_COUNTER += 1
                        yawn_start_time = time.time()
                        cv2.putText(frame, str(YAWN_COUNTER), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        score = max(score - penalty_yawning, 0)
                    if YAWN_COUNTER >= 2:
                        alert_message = "Alert! You have yawned more than twice. Do you want to listen to some music?"
                        speak_alert(alert_message)
                        command = get_voice_command()
                        if command:
                            if "yes" in command.lower():
                                subprocess.call('Spotify.exe')
                            elif "no" in command.lower():
                                print("User declined to listen to music.")
                            else:
                                print("Unrecognized command.")
                        yawn_alert_spoken = True
                        YAWN_COUNTER = 0
        else:
            yawn_in_progress = False
            yawn_start_time = None
            yawn_alert_spoken = False
            cv2.putText(frame, "NOT YAWNING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    current_time = time.time()
    if current_time - last_score_update >= score_update_interval:
        last_score_update = current_time
        cv2.putText(frame, f"Score: {score}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        print(f"Updated Score: {score}")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Example user info update
email = "user@example.com"
user_rating = 10
update_user_field(email, 'rating', user_rating + score)

vs.release()
cv2.destroyAllWindows()
        
