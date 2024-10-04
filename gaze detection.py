import cv2
import numpy as np
from openvino.runtime import Core, Tensor
from scipy.spatial import distance as dist

# Load OpenVINO runtime and models
core = Core()

# Paths to downloaded models
face_detection_model_path = r"F:\one api packages\intel\face-detection-adas-0001\FP32\face-detection-adas-0001.xml"
head_pose_model_path = r"F:\one api packages\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml"
landmarks_model_path = r"F:\one api packages\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml"
gaze_model_path = r"F:\one api packages\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml"

# Load models
face_detection_model = core.read_model(face_detection_model_path)
head_pose_model = core.read_model(head_pose_model_path)
landmarks_model = core.read_model(landmarks_model_path)
gaze_model = core.read_model(gaze_model_path)

# Compile models to create ExecutableNetworks
face_executable_network = core.compile_model(face_detection_model, "CPU")
head_pose_executable_network = core.compile_model(head_pose_model, "CPU")
landmarks_executable_network = core.compile_model(landmarks_model, "CPU")
gaze_executable_network = core.compile_model(gaze_model, "CPU")

# Get input and output layers for the models
face_input_layer = face_executable_network.input(0)
face_output_layer = face_executable_network.output(0)

landmarks_input_layer = landmarks_executable_network.input(0)
landmarks_output_layer = landmarks_executable_network.output(0)

# Initialize laptop webcam for real-time video capture
cap = cv2.VideoCapture(1)

# Function to preprocess input frames for the models
def preprocess_frame(frame, input_size):
    input_image = cv2.resize(frame, input_size)  # Resize frame
    input_image = input_image.transpose(2, 0, 1)  # Change from HWC to CHW
    input_image = input_image[np.newaxis, ...]  # Add batch dimension
    input_image = input_image.astype(np.float32)  # Convert to float32 for the model
    return input_image

# Function to analyze eye region intensity for open/closed detection
def is_eye_closed(eye_region):
    if eye_region.size == 0:
        return False  # If the eye region is empty, assume it's open

    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)  # Simple threshold
    non_zero_pixels = cv2.countNonZero(thresh_eye)  # Count non-zero pixels in the eye region
    eye_area = gray_eye.size

    # If most of the eye area is non-zero (bright pixels), assume eyes are open, else closed
    if non_zero_pixels / eye_area < 0.5:
        return True  # Eyes are closed
    return False  # Eyes are open

# Create InferRequests for face detection, head pose estimation, and gaze detection
face_infer_request = face_executable_network.create_infer_request()
landmarks_infer_request = landmarks_executable_network.create_infer_request()

# Real-time head pose detection and gaze monitoring loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from laptop camera.")
        break

    # Preprocess the frame for face detection
    face_input_image = preprocess_frame(frame, (672, 384))  # Model-specific input size

    # Convert the NumPy array to OpenVINO Tensor
    face_input_tensor = Tensor(face_input_image)

    # Set input tensor for face detection
    face_infer_request.set_input_tensor(0, face_input_tensor)

    # Perform face detection inference
    face_infer_request.infer()

    # Get the output tensor for face detection
    face_output_tensor = face_infer_request.get_output_tensor(face_output_layer.index)
    faces = face_output_tensor.data  # Access the raw data from the tensor

    # Loop through detected faces
    if faces.shape[2] > 0:  # Check if any faces were detected
        for face in faces[0][0]:
            confidence = face[2]
            if confidence > 0.5:  # Confidence threshold
                xmin, ymin, xmax, ymax = (face[3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)

                # Clamp the coordinates to the frame boundaries
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(frame.shape[1], xmax)
                ymax = min(frame.shape[0], ymax)

                # Extract the face ROI with clamped coordinates
                face_roi = frame[ymin:ymax, xmin:xmax]

                # Check if face_roi is valid before further processing
                if face_roi.size == 0:
                    print("Face region is empty, skipping this frame.")
                    continue

                # Preprocess the detected face region for landmarks detection
                landmarks_input_image = preprocess_frame(face_roi, (48, 48))  # Model input size for landmarks

                # Convert the NumPy array to OpenVINO Tensor
                landmarks_input_tensor = Tensor(landmarks_input_image)

                # Set input tensor for landmarks detection
                landmarks_infer_request.set_input_tensor(0, landmarks_input_tensor)

                # Perform landmarks detection
                landmarks_infer_request.infer()

                # Get the output tensor for landmarks
                landmarks_output = landmarks_infer_request.get_output_tensor(landmarks_output_layer.index).data

                # The landmarks output contains 5 points: [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
                landmarks = landmarks_output.reshape(-1, 2)

                # Get eye coordinates (left_eye and right_eye)
                left_eye = landmarks[0]  # x, y coordinates for the left eye center
                right_eye = landmarks[1]  # x, y coordinates for the right eye center

                # Ensure the eye coordinates are in the right range
                left_eye_x, left_eye_y = int(left_eye[0] * face_roi.shape[1] + xmin), int(left_eye[1] * face_roi.shape[0] + ymin)
                right_eye_x, right_eye_y = int(right_eye[0] * face_roi.shape[1] + xmin), int(right_eye[1] * face_roi.shape[0] + ymin)

                # Draw circles at the eye positions for visualization
                cv2.circle(frame, (left_eye_x, left_eye_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (right_eye_x, right_eye_y), 5, (0, 255, 0), -1)

                # Extract eye regions for open/closed detection (30x30 region around each eye)
                left_eye_region = frame[left_eye_y-15:left_eye_y+15, left_eye_x-15:left_eye_x+15]
                right_eye_region = frame[right_eye_y-15:right_eye_y+15, right_eye_x-15:right_eye_x+15]

                # Check if eyes are closed
                left_eye_closed = is_eye_closed(left_eye_region)
                right_eye_closed = is_eye_closed(right_eye_region)

                # Display result based on eye status
                if left_eye_closed and right_eye_closed:
                    cv2.putText(frame, "Eyes Closed", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Eyes Open", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output frame with all annotations
    cv2.imshow('Head Pose and Gaze Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
