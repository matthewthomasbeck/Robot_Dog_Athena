import cv2
import subprocess
import numpy as np
from openvino.runtime import Core
import signal
import sys
import time

def signal_handler(sig, frame):
    print("\nGracefully exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    # Load smaller model for testing
    model_path = "/home/matthewthomasbeck/Projects/Robot_Dog/model/mobilenet_ssd.xml"
    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
    output_layer = compiled_model.outputs[0]

    # Start rpicam-vid process
    camera_process = subprocess.Popen(
        ["rpicam-vid", "--width", "640", "--height", "480", "--framerate", "10", "--timeout", "0", "--output", "-", "--codec", "mjpeg", "--nopreview"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    mjpeg_buffer = b''
    last_inference_time = 0
    desired_fps = 5

    try:
        while True:
            chunk = camera_process.stdout.read(8192)
            if not chunk:
                print("Camera process stopped sending data.")
                break

            mjpeg_buffer += chunk
            start_idx = mjpeg_buffer.find(b'\xff\xd8')
            end_idx = mjpeg_buffer.find(b'\xff\xd9')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                frame_data = mjpeg_buffer[start_idx:end_idx + 2]
                mjpeg_buffer = mjpeg_buffer[end_idx + 2:]
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                if time.time() - last_inference_time < 1.0 / desired_fps:
                    continue

                last_inference_time = time.time()
                input_image = cv2.resize(frame, (300, 300)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
                results = compiled_model([input_image])[output_layer]

                # Process detections
                detections = results[0][0]
                for detection in detections:
                    conf = float(detection[2])
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, detection[3:7] * [640, 480, 640, 480])
                        label = f"{conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Video Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
    finally:
        cv2.destroyAllWindows()
        camera_process.terminate()
        camera_process.wait()

if __name__ == "__main__":
    main()

