import cv2
import numpy as np
from openvino.runtime import Core

# Load OpenVINO Runtime
ie = Core()

# Load the face detection model (IR files: .xml and .bin)
model_xml = r"C:/Users/marut/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
model_bin = r"C:\Users\marut\intel\face-detection-adas-0001\FP32\face-detection-adas-0001.bin"

# Load the network and get input/output layer info
net = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=net, device_name="CPU")

# Get input and output layer information
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Open the camera or video file (use 0 for a webcam)
cap = cv2.VideoCapture(0)

# Define the size for model input
input_height, input_width = input_layer.shape[2], input_layer.shape[3]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for inference (resize and transpose)
    input_image = cv2.resize(frame, (input_width, input_height))
    input_image = input_image.transpose((2, 0, 1))  # HWC to CHW format
    input_image = np.expand_dims(input_image, axis=0)

    # Perform inference
    result = compiled_model([input_image])[output_layer]

    # Parse the detection results
    for detection in result[0][0]:
        # Detection format: [image_id, label, confidence, xmin, ymin, xmax, ymax]
        confidence = detection[2]
        if confidence > 0.5:  # Set confidence threshold
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Face Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
