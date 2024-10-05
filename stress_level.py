import cv2
import numpy as np
from openvino.runtime import Core

# Load OpenVINO Runtime
ie = Core()

# Paths to the model files (adjust these paths according to your system)
model_xml = r"C:\Users\marut\intel\emotions-recognition-retail-0003\FP32\emotions-recognition-retail-0003.xml"
model_bin = r"C:\Users\marut\intel\emotions-recognition-retail-0003\FP32\emotions-recognition-retail-0003.bin"

# Read the model
net = ie.read_model(model=model_xml, weights=model_bin)

# Compile the model for CPU
compiled_model = ie.compile_model(model=net, device_name="CPU")

# Get input and output layers of the model
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Emotion labels for the output from the model
emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# Open the webcam
cap = cv2.VideoCapture(0)

# Get the input shape of the model
n, c, h, w = input_layer.shape

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input image for the network (resize and transpose)
    input_image = cv2.resize(frame, (w, h))
    input_image = input_image.transpose((2, 0, 1))  # HWC to CHW format
    input_image = np.expand_dims(input_image, axis=0)

    # Perform inference
    result = compiled_model([input_image])[output_layer]

    # Get the emotion with the highest probability
    emotion_index = np.argmax(result[0])
    emotion_label = emotions[emotion_index]

    # Display the emotion on the frame
    cv2.putText(frame, f"Emotion: {emotion_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Emotion Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
