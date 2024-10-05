import firebase_admin
from firebase_admin import credentials, db
import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
import time
import pyttsx3  # Importing TTS library
from openvino.runtime import Core  # Updated OpenVINO library
import subprocess
import speech_recognition as sr


def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()  # Return the command in lowercase for consistency
    except sr.UnknownValueError:
        print("Sorry, I did not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
# Initialize Firebase app with service account credentials

# Initialize TTS engine
engine = pyttsx3.init()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # Vertical distance
    B = dist.euclidean(mouth[4], mouth[8])   # Vertical distance
    C = dist.euclidean(mouth[0], mouth[6])   # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar

def speak_alert(alert_message):
    engine.say(alert_message)
    engine.runAndWait()

# Thresholds for eye and yawn detection
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.75  # Adjusted threshold for yawning
EYE_CLOSED_TIME_THRESH = 3  # Time threshold for alert in seconds
YAWN_TIME_THRESH = 2.5  # Yawn must last at least 3 seconds to be counted

# Load OpenVINO face detection model (for the box)
print("-> Loading OpenVINO face detection model...")
ie = Core()
model_xml = r"C:\Windows\System32\intel\face-detection-adas-0001\FP32\face-detection-adas-0001.xml"
compiled_model = ie.compile_model(model=model_xml, device_name="CPU")
input_layer = compiled_model.input(0)

# Load dlib's shape predictor for facial landmarks
print("-> Loading the predictor...")
shape_predictor_path = r"C:\Users\Mahesh\Downloads\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(shape_predictor_path)

# Start the video stream
print("-> Starting the video stream...")
vs = cv2.VideoCapture(1)
time.sleep(1.0)

COUNTER = 0
YAWN_COUNTER = 0
score = 100  # Initial score
penalty_eye_closure = 0.22  # Penalty for prolonged eye closure
penalty_yawning = 0.11  # Penalty for yawning
last_score_update = time.time()
score_update_interval = 1200  # Update score every 20 minutes

eye_alert_spoken = False
yawn_alert_spoken = False
yawn_in_progress = False  # Track if a yawn is currently in progress

# Buffers for smoothing MAR
mar_buffer = []
buffer_size = 10  # Size of buffer to smooth MAR

start_time = None  # Initialize variable for timing eye closure
yawn_start_time = None  # Initialize variable for timing yawning

while True:
    ret, frame = vs.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Prepare the frame for OpenVINO face detection
    n, c, h, w = input_layer.shape
    p_frame = cv2.resize(frame, (w, h))
    p_frame = p_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    p_frame = p_frame.reshape(n, c, h, w)

    # Perform inference to detect faces (for drawing the box)
    result = compiled_model([p_frame])[compiled_model.output(0)]
    face_detections = result[0][0]

    for detection in face_detections:
        confidence = detection[2]
        if confidence > 0.5:
            # Extract face coordinates for drawing the box
            x_min = int(detection[3] * frame.shape[1])
            y_min = int(detection[4] * frame.shape[0])
            x_max = int(detection[5] * frame.shape[1])
            y_max = int(detection[6] * frame.shape[0])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Detect facial landmarks using dlib predictor
            rect = dlib.rectangle(x_min, y_min, x_max, y_max)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract eye and mouth coordinates
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

            # Smooth MAR
            mar_buffer.append(mar)
            if len(mar_buffer) > buffer_size:
                mar_buffer.pop(0)
            smooth_mar = sum(mar_buffer) / len(mar_buffer)

            # Visualize eyes and mouth
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

            # Check for eye closure
            if ear < EYE_AR_THRESH:
                if start_time is None:
                    start_time = time.time()
                COUNTER += 1
                elapsed_time = time.time() - start_time

                if elapsed_time >= EYE_CLOSED_TIME_THRESH and not eye_alert_spoken:
                    alert_message = "Alert! Eyes closed for more than 4 seconds."
                    cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    speak_alert(alert_message)  # Trigger TTS alert
                    eye_alert_spoken = True
                    score = max(score - penalty_eye_closure, 0)  # Apply penalty for eye closure
                    alert_message = "Are you feeling drowsy? Do you want to listen to some music?"
                    speak_alert(alert_message)  # Trigger TTS alert for yawning
                    # Assuming get_voice_command() is defined elsewhere
                    command = get_voice_command()  # Get voice command from user
                    if command:
                        if "yes" in command:
                            subprocess.call('Spotify.exe')
                        elif "no" in command:
                            print("User declined to listen to music.")
                        else:
                            print("Unrecognized command.")
            else:
                COUNTER = 0
                start_time = None
                eye_alert_spoken = False
                cv2.putText(frame, "EYES OPEN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check for yawning
            if mar > MOUTH_AR_THRESH:
                if not yawn_in_progress:
                    yawn_start_time = time.time()  # Start timing the yawn
                    yawn_in_progress = True
                else:
                    yawn_duration = time.time() - yawn_start_time

                    # Only count the yawn if it lasts longer than the threshold
                    if yawn_duration >= YAWN_TIME_THRESH and not yawn_alert_spoken:
                        alert_message = "Alert! Yawning detected. Please stay alert."
                        cv2.putText(frame, alert_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        speak_alert(alert_message)  # Trigger TTS alert for yawning
                        yawn_alert_spoken = True
                        score = max(score - penalty_yawning, 0)  # Apply penalty for yawning
            else:
                yawn_in_progress = False
                yawn_alert_spoken = False  # Reset yawn alert state

    # Display the score
    cv2.putText(frame, f"Score: {score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Check if enough time has passed to update the score in Firebase
    current_time = time.time()
    if current_time - last_score_update >= score_update_interval:
      # Update the score in Firebase
        last_score_update = current_time
  
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
print(score)
# Cleanup
vs.release()
cv2.destroyAllWindows()
