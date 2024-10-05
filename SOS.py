import cv2
import mediapipe as mp
import requests
from openvino.runtime import Core

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

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

    if index_distance < 0.1 and middle_distance < 0.1 and ring_distance < 0.1 and pinky_distance < 0.1:
        return True
    return False

def is_peace_sign(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    return (index_tip < middle_tip < thumb_tip and 
            ring_tip > middle_tip and 
            pinky_tip > middle_tip and 
            abs(thumb_tip - ring_tip) > 0.05 and 
            abs(thumb_tip - pinky_tip) > 0.05)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_clenched_fist(hand_landmarks.landmark):
                print("Clenched fist detected! Stopping the program...")
                cap.release()
                cv2.destroyAllWindows()
                exit()  # Exit the program

            elif is_peace_sign(hand_landmarks.landmark):
                print("Peace sign detected! Stopping the program...")
                cap.release()
                cv2.destroyAllWindows()
                exit()  # Exit the program

            # Draw landmarks with a specified color (e.g., Red in BGR format)
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, 
                                                       mp_hands.HAND_CONNECTIONS, 
                                                       mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    cv2.imshow("Hand Gesture Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for display

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
