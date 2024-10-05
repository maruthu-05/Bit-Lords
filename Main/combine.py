def model(email):
        import cv2
        import mediapipe as mp
        import dlib
        import numpy as np
        from scipy.spatial import distance as dist
        import time
        import pyttsx3
        import firebase_admin
        from firebase_admin import credentials, db
        from openvino.runtime import Core
        import subprocess
        import speech_recognition as sr
        from imutils import face_utils
        
        # Initialize Mediapipe for hand tracking
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        
        # Initialize Firebase app with service account credentials
        cred = credentials.Certificate(r"C:\Users\Mahesh\Downloads\cour (2) (1)\cour\dms-hackthon-firebase-adminsdk-508l6-991963bae2.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://dms-hackthon-default-rtdb.firebaseio.com/"
        })
        
        # Initialize TTS engine
        engine = pyttsx3.init()
        
        # OpenVINO face detection model
        print("-> Loading OpenVINO face detection model...")
        ie = Core()
        model_xml = r"C:\Windows\System32\intel\face-detection-adas-0001\FP32\face-detection-adas-0001.xml"
        compiled_model = ie.compile_model(model=model_xml, device_name="CPU")
        input_layer = compiled_model.input(0)
        
        # Load dlib's shape predictor for facial landmarks
        print("-> Loading the predictor...")
        shape_predictor_path = r"C:\Users\Mahesh\Downloads\shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(shape_predictor_path)
        
        # Thresholds and buffers
        EYE_AR_THRESH = 0.22
        MOUTH_AR_THRESH = 0.75
        EYE_CLOSED_TIME_THRESH = 3
        YAWN_TIME_THRESH = 2.5
        penalty_eye_closure = 0.22
        penalty_yawning = 0.11
        score_update_interval = 1200
        COUNTER = 0
        YAWN_COUNTER = 0
        score = 100
        eye_alert_spoken = False
        yawn_alert_spoken = False
        yawn_in_progress = False
        mar_buffer = []
        buffer_size = 10
        start_time = None
        yawn_start_time = None
        
        # Gesture detection functions
        def is_clenched_fist(landmarks):
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
            index_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            middle_distance = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
            ring_distance = ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5
            pinky_distance = ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5
            return index_distance < 0.1 and middle_distance < 0.1 and ring_distance < 0.1 and pinky_distance < 0.1
        
        def is_peace_sign(landmarks):
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
            return (index_tip < middle_tip < thumb_tip and ring_tip > middle_tip and pinky_tip > middle_tip and
                    abs(thumb_tip - ring_tip) > 0.05 and abs(thumb_tip - pinky_tip) > 0.05)
        
        # Eye and Mouth Aspect Ratios
        def eye_aspect_ratio(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C)
            return ear
        
        def mouth_aspect_ratio(mouth):
            A = dist.euclidean(mouth[2], mouth[10])
            B = dist.euclidean(mouth[4], mouth[8])
            C = dist.euclidean(mouth[0], mouth[6])
            mar = (A + B) / (2.0 * C)
            return mar
        
        def speak_alert(alert_message):
            engine.say(alert_message)
            engine.runAndWait()
        
        def update_user_field(email, field_name, new_value):
            ref = db.reference('registerform')
            users = ref.order_by_child('Email').equal_to(email).get()
            if users:
                for user_id, user_info in users.items():
                    ref.child(user_id).update({field_name: new_value})
                    print(f"Updated {field_name} for {email} to {new_value}.")
                    return
            print(f"Failed to update: No user found with the email {email}.")
        
        def get_voice_command():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening for a command...")
                audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio)
                print(f"You said: {command}")
                return command.lower()
            except sr.UnknownValueError:
                print("Sorry, I did not understand the audio.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return None
        
        # Start video capture
        cap = cv2.VideoCapture(1)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
        
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
        
            # Hand gesture detection
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_clenched_fist(hand_landmarks.landmark):
                        print("Clenched fist detected! Stopping the program...")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                    elif is_peace_sign(hand_landmarks.landmark):
                        print("Peace sign detected! Stopping the program...")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
            # Face detection and facial landmarks detection
            n, c, h, w = input_layer.shape
            p_frame = cv2.resize(frame, (w, h)).transpose((2, 0, 1)).reshape(n, c, h, w)
            result = compiled_model([p_frame])[compiled_model.output(0)]
            face_detections = result[0][0]
        
            for detection in face_detections:
                confidence = detection[2]
                if confidence > 0.5:
                    x_min = int(detection[3] * frame.shape[1])
                    y_min = int(detection[4] * frame.shape[0])
                    x_max = int(detection[5] * frame.shape[1])
                    y_max = int(detection[6] * frame.shape[0])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    rect = dlib.rectangle(x_min, y_min, x_max, y_max)
                    shape = predictor(gray, rect)
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
        
                    mar_buffer.append(mar)
                    if len(mar_buffer) > buffer_size:
                        mar_buffer.pop(0)
                    smooth_mar = sum(mar_buffer) / len(mar_buffer)
        
                    # Visualize eyes and mouth
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
                    # Eye closure detection
                    if ear < EYE_AR_THRESH:
                        if start_time is None:
                            start_time = time.time()
                        elif time.time() - start_time >= EYE_CLOSED_TIME_THRESH:
                            if not eye_alert_spoken:
                                speak_alert("Your eyes are closed. Please stay alert!")
                                eye_alert_spoken = True
                            score -= penalty_eye_closure
                            update_user_field('driver@example.com', 'score', score)
                    else:
                        start_time = None
                        eye_alert_spoken = False
        
                    # Yawning detection
                    if smooth_mar > MOUTH_AR_THRESH:
                        if not yawn_in_progress:
                            yawn_start_time = time.time()
                            yawn_in_progress = True
                        elif time.time() - yawn_start_time >= YAWN_TIME_THRESH:
                            if not yawn_alert_spoken:
                                speak_alert("You seem to be yawning. Please take a break!")
                                yawn_alert_spoken = True
                            score -= penalty_yawning
                            update_user_field('driver@example.com', 'score', score)
                    else:
                        yawn_in_progress = False
                        yawn_alert_spoken = False
        
            # Display the frame
            cv2.imshow('Driver Behavior Monitoring System', frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
