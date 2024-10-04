def model(email):
    import firebase_admin
    from firebase_admin import credentials, db
    import cv2
    import subprocess
    import dlib
    from imutils import face_utils
    import numpy as np
    from scipy.spatial import distance as dist
    import time
    import pyttsx3  # Importing TTS library
    import speech_recognition as sr  # Importing speech recognition library


    # Global variable to store the rating value
    user_rating = 0

    # Initialize the Firebase app with service account credentials
    cred = credentials.Certificate(r"C:\Users\Mahesh\Downloads\cour (2) (1)\cour\dms-hackthon-firebase-adminsdk-508l6-991963bae2.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL':"https://dms-hackthon-default-rtdb.firebaseio.com/"  # Use the correct database URL without specific user reference
    })

    def get_user_rating(email):
        global user_rating
        # Reference to the "registerform" collection
        ref = db.reference('registerform')

        # Query the database for the user with the matching email
        users = ref.order_by_child('Email').equal_to(email).get()

        if users:
            # Assuming there is only one user with that email
            for user_id, user_info in users.items():
                user_rating = user_info.get('rating', None)  # Get the rating, default to None if not found
                print(f"User rating for {email}: {user_rating}")
                return user_id
        else:
            print(f"No user found with the email: {email}")
            return None

    def update_user_field(email, field_name, new_value):
        # Get the user_id by querying with the email
        user_id = get_user_rating(email)

        if user_id:
            # Reference to the specific user in "registerform"
            try:
                ref = db.reference(f'registerform/{user_id}')
                ref.update({field_name: new_value})
                print(f"Updated {field_name} for {email} to {new_value}.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print(f"Failed to update: No user found with the email {email}.")


    # Initialize TTS engine
    engine = pyttsx3.init()

    # Functions to calculate eye and mouth aspect ratios
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

    # Function to handle TTS alerts
    def speak_alert(alert_message):
        engine.say(alert_message)
        engine.runAndWait()

    # Function to get voice command
    def get_voice_command():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening for your response...")
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")
                return command
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
                return None
            except sr.RequestError:
                print("Could not request results; check your network connection.")
                return None

    # Thresholds for eye and yawn detection
    EYE_AR_THRESH = 0.22
    MOUTH_AR_THRESH = 0.7865  # Threshold for yawning
    EYE_CLOSED_TIME_THRESH = 3  # Time threshold for alert in seconds
    YAWN_TIME_THRESH = 2.5  # Yawn must last at least 3 seconds to be counted

    # Initialize dlib's face detector and facial landmark predictor
    print("-> Loading the predictor and detector...")
    shape_predictor_path = r"C:\Users\Mahesh\Downloads\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Start the video stream
    print("-> Starting the video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)

    COUNTER = 0
    YAWN_COUNTER = 0
    start_time = None
    yawn_start_time = None
    score = 100  # Initial score
    penalty_eye_closure = 0.22  # Penalty for prolonged eye closure
    penalty_yawning = 0.11  # Penalty for yawning

    # Time-related variables for score updates
    last_score_update = time.time()
    score_update_interval = 1200  # Update score every 60 seconds

    eye_alert_spoken = False
    yawn_alert_spoken = False
    yawn_in_progress = False  # Track if a yawn is currently in progress

    while True:
        ret, frame = vs.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract the eye and mouth coordinates
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

            # Visualize eyes and mouth
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # Check for eye closure
            if ear < EYE_AR_THRESH:
                if start_time is None:
                    start_time = time.time()
                COUNTER += 1
                elapsed_time = time.time() - start_time

                if elapsed_time >= EYE_CLOSED_TIME_THRESH and not eye_alert_spoken:
                    alert_message = "Alert! Eyes closed for more than 4 seconds."
                    cv2.putText(frame, alert_message, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    speak_alert(alert_message)  # Trigger TTS alert
                    eye_alert_spoken = True
                    score = max(score - penalty_eye_closure, 0)  # Apply penalty for eye closure
                    alert_message = "Are you feeling drowsy? Do you want to listen to some music?"
                    speak_alert(alert_message)  # Trigger TTS alert for yawning
                    command = get_voice_command()  # Get voice command from user
                    if command:
                        if "yes" in command:
                            subprocess.call('Spotify.exe')
                        elif "no" in command:
                            print("User declined to listen to music.")
                        else:
                            print("Unrecognized command.")
                    yawn_alert_spoken = True  # Prevent multiple alerts
                    YAWN_COUNTER = 0  # Reset yawn counter after the alert
            else:
                COUNTER = 0
                start_time = None
                eye_alert_spoken = False
                cv2.putText(frame, "EYES OPEN", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check for yawning
            if mar > MOUTH_AR_THRESH:
                if not yawn_in_progress:
                    yawn_start_time = time.time()  # Start timing the yawn
                    yawn_in_progress = True
                else:
                    yawn_duration = time.time() - yawn_start_time

                    # Only count the yawn if it lasts longer than 3 seconds
                    if yawn_duration >= 2:
                        cv2.putText(frame, "YAWNING", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if not yawn_alert_spoken:
                            YAWN_COUNTER += 1
                            yawn_start_time = time.time()
                            cv2.putText(frame, str(YAWN_COUNTER), (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            score = max(score - penalty_yawning, 0)
                        if YAWN_COUNTER >= 2:
                            alert_message = "Alert! You have yawned more than twice. Do you want to listen to some music?"
                            speak_alert(alert_message)  # Trigger TTS alert for yawning
                            command = get_voice_command()  # Get voice command from user
                            if command:
                                if "yes" in command:
                                    subprocess.call('Spotify.exe')
                                elif "no" in command:
                                    print("User declined to listen to music.")
                                else:
                                    print("Unrecognized command.")
                            yawn_alert_spoken = True  # Prevent multiple alerts
                            YAWN_COUNTER = 0  # Reset yawn counter after the alert
                
            else:
                # Reset yawn-related variables if no yawn is detected
                yawn_in_progress = False
                yawn_start_time = None
                yawn_alert_spoken = False
                cv2.putText(frame, "NOT YAWNING", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update the score every minute
        current_time = time.time()
        if current_time - last_score_update >= score_update_interval:
            last_score_update = current_time
            cv2.putText(frame, f"Score: {score}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            print(f"Updated Score: {score}")
            

        # Display the video frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    # Example usage

    email_to_update = email  # Email to check
    field_to_update = 'rating'  # Field to update
    new_value = user_rating+score

    # Get user rating and update the field
    update_user_field(email_to_update, field_to_update, new_value)
    # Clean up and close the video stream

    vs.release()
    cv2.destroyAllWindows()

