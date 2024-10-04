import cv2
import numpy as np
from openvino.runtime import Core

# Load the OpenVINO model
ie = Core()

# Model paths
model_paths = {
    "person_detection": r"C:\Downloads\Models\intel\person-detection-asl-0001\FP32\person-detection-asl-0001.xml",
    "asl_recognition": r"C:\Downloads\Models\intel\asl-recognition-0004\FP32\asl-recognition-0004.xml",
    "sign_language": r"C:\Downloads\Models\intel\common-sign-language-0002\FP32\common-sign-language-0002.xml"
}

weights_paths = {
    "person_detection": r"C:\Downloads\Models\intel\person-detection-asl-0001\FP32\person-detection-asl-0001.bin",
    "asl_recognition": r"C:\Downloads\Models\intel\asl-recognition-0004\FP32\asl-recognition-0004.bin",
    "sign_language": r"C:\Downloads\Models\intel\common-sign-language-0002\FP32\common-sign-language-0002.bin"
}

# Load and compile the models
exec_nets = {name: ie.read_model(model=model_path, weights=weights_paths[name]) for name, model_path in model_paths.items()}
exec_nets = {name: ie.compile_model(model=exec_net, device_name="CPU") for name, exec_net in exec_nets.items()}

# Load webcam for capturing video
cap = cv2.VideoCapture(0)

# Buffer for storing frames for temporal input
frame_buffer = []
max_frames = 8  # Assuming the model requires 8 frames

# Gesture detection function
def detect_gesture(gesture_ids):
    """
    Detect specific gestures based on gesture IDs.
    Here we check for the signs of Help, Ambulance, and Police.
    """
    if len(gesture_ids) == 0:
        return None

    # Debug: Print the gesture IDs detected by the model
    print(f"Detected gesture IDs: {gesture_ids}")

    # Check for specific gesture IDs (Assuming these gesture IDs are the correct ones)
    if 1 in gesture_ids and 2 in gesture_ids:  # Thumb and Index raised
        return "HELP"
    elif 1 in gesture_ids and 5 in gesture_ids:  # Thumb and Pinky raised
        return "AMBULANCE"
    elif 1 in gesture_ids:  # Only Thumb raised
        return "POLICE"
    
    return None

# Flag for the last detected gesture
last_detected_gesture = None

# Main loop for gesture detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input for person detection
    input_image_pd = cv2.resize(frame, (320, 320))  # Resize to 320x320 as expected by the person detection model
    input_image_pd = input_image_pd.transpose((2, 0, 1))  # Change data layout to C,H,W
    input_image_pd = np.expand_dims(input_image_pd, axis=0)  # Add batch dimension

    # Get input name and perform person detection
    input_name_pd = next(iter(exec_nets["person_detection"].inputs))
    results_pd = exec_nets["person_detection"]({input_name_pd: input_image_pd})
    output_node_pd = next(iter(exec_nets["person_detection"].outputs))
    person_detections = results_pd[output_node_pd]

    # Adjust detection logic based on the actual output format
    person_detected = np.any(person_detections)  # Check if any person is detected

    if not person_detected:
        cv2.imshow('Hand Gesture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Prepare input for gesture recognition
    input_image_gesture = cv2.resize(frame, (224, 224))  # Resize to 224x224 as expected by the gesture recognition model
    input_image_gesture = input_image_gesture.transpose((2, 0, 1))  # Change data layout to C,H,W
    input_image_gesture = np.expand_dims(input_image_gesture, axis=0)  # Add batch dimension

    # Append to frame buffer and process only if enough frames are collected
    frame_buffer.append(input_image_gesture)
    if len(frame_buffer) > max_frames:
        frame_buffer.pop(0)

    if len(frame_buffer) == max_frames:
        # Stack frames to create a 5D tensor (1, 3, 8, 224, 224)
        input_tensor = np.concatenate(frame_buffer, axis=0)  # Shape: (8, 3, 224, 224)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Shape: (1, 8, 3, 224, 224)
        input_tensor = input_tensor.transpose(0, 2, 1, 3, 4)  # Shape: (1, 3, 8, 224, 224)

        # Perform gesture recognition
        input_name_gesture = next(iter(exec_nets["sign_language"].inputs))
        results_gesture = exec_nets["sign_language"]({input_name_gesture: input_tensor})
        output_node_gesture = next(iter(exec_nets["sign_language"].outputs))
        gesture_ids = results_gesture[output_node_gesture].astype(int).flatten()  # Assuming model returns gesture IDs

        # Detect gesture
        detected_gesture = detect_gesture(gesture_ids)

        if detected_gesture and detected_gesture != last_detected_gesture:
            last_detected_gesture = detected_gesture
            alert_text = f"Alert: {detected_gesture} Needed!"
            cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(alert_text)

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
