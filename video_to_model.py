import sys
import cv2
import numpy as np
from openvino.runtime import Core

def main():
    width = 2304
    height = 1296

    # Initialize OpenVINO runtime and load the face detection model from your project directory
    ie = Core()
    model_path = "/home/matthewthomasbeck/Projects/Robot_Dog/model/model.yml"
    
    print("MODEL FOUND")

    compiled_model = ie.compile_model(model=model_path, device_name="MYRIAD")

    # Get the input and output layers
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    while True:
        # Read MJPEG frame from stdin
        raw_image = sys.stdin.buffer.read(width * height * 2)

        if len(raw_image) != width * height * 2:
            print("Incomplete frame received")
            break

        # Decode MJPEG frame
        frame = np.frombuffer(raw_image, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode frame")
            break

        # Prepare the frame for the model (preprocessing)
        resized_frame = cv2.resize(frame, (input_layer.shape[3], input_layer.shape[2]))
        input_data = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)

        # Perform inference
        results = compiled_model([input_data])[output_layer]

        # Post-process the results and draw bounding boxes
        for result in results[0][0]:
            confidence = result[2]
#            if confidence > 0.01:  # Filter weak detections
            xmin = int(result[3] * frame.shape[1])
            ymin = int(result[4] * frame.shape[0])
            xmax = int(result[5] * frame.shape[1])
            ymax = int(result[6] * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Display the frame with detection results
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

