import cv2
import numpy as np
import subprocess
from pathlib import Path
from ultralytics import YOLO

# Define model path
MODEL_FOLDER = Path("~/Projects/Robot_Dog/model/").expanduser()
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_FOLDER / "yolov8n.pt"

# Ensure YOLOv8 model is downloaded
if not MODEL_PATH.exists():
    print(f"Downloading YOLOv8 model to {MODEL_PATH}...")
    yolo_model = YOLO("yolov8n")
    yolo_model.overrides["save_dir"] = str(MODEL_FOLDER)  # Save to model directory
    yolo_model.export(format="torchscript")  # Ensure compatibility
    yolo_model = YOLO(str(MODEL_PATH))  # Reload from disk
else:
    yolo_model = YOLO(str(MODEL_PATH))  # Load existing model

# Function to detect objects in a frame
def detect_objects(frame):
    """Run YOLOv8 detection on a frame."""
    results = yolo_model.predict(source=frame, save=False, save_txt=False, conf=0.25)
    return results

def main():
    # Start the rpicam-vid process
    camera_process = subprocess.Popen(
        ["rpicam-vid", "--width", "640", "--height", "480", "--framerate", "30", "--timeout", "0", "--output", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    mjpeg_buffer = b''

    try:
        while True:
            # Read MJPEG data
            chunk = camera_process.stdout.read(4096)
            if not chunk:
                print("Camera process stopped sending data.")
                break

            mjpeg_buffer += chunk

            # Find JPEG frame markers
            start_idx = mjpeg_buffer.find(b'\xff\xd8')
            end_idx = mjpeg_buffer.find(b'\xff\xd9')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # Extract and decode frame
                frame_data = mjpeg_buffer[start_idx:end_idx + 2]
                mjpeg_buffer = mjpeg_buffer[end_idx + 2:]
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # Run YOLOv8 detection
                    results = detect_objects(frame)

                    # Draw results on the frame
                    for result in results[0].boxes:
                        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                        conf = result.conf[0].item()
                        label = f"{result.cls[0]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display the frame
                    cv2.imshow("Robot Camera - YOLOv8", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Failed to decode frame, skipping...")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera_process.terminate()
        camera_process.wait()

if __name__ == "__main__":
    main()

