import cv2
import numpy as np
from openvino.runtime import Core

# Load OpenVINO Runtime
ie = Core()

# Path to the head pose model files (adjust paths as necessary)
model_xml = r"C:\Users\marut\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml"
model_bin = r"C:\Users\marut\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.bin"

# Read the model
net = ie.read_model(model=model_xml, weights=model_bin)

# Compile the model for the CPU
compiled_model = ie.compile_model(model=net, device_name="CPU")

# Get input and output layers of the model
input_layer = compiled_model.input(0)
output_layers = [compiled_model.output(i) for i in range(3)]  # Yaw, Pitch, Roll

# Video capture (from webcam or video file)
cap = cv2.VideoCapture(0)

# Input shape from the model
n, c, h, w = input_layer.shape

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit the model's expected input size
    resized_frame = cv2.resize(frame, (w, h))

    # Prepare the frame as model input
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

    # Perform inference to get yaw, pitch, and roll angles
    results = compiled_model([input_image])

    yaw = results[output_layers[0]][0][0]
    pitch = results[output_layers[1]][0][0]
    roll = results[output_layers[2]][0][0]

    # Display the head pose angles on the frame
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Head Pose Estimation", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
