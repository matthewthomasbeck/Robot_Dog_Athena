import cv2
import subprocess
import numpy as np

def main():
    # Start the rpicam-vid process
    camera_process = subprocess.Popen(
        ["rpicam-vid", "--width", "640", "--height", "480", "--framerate", "30", "--timeout", "0", "--output", "-", "--codec", "mjpeg"],
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
                    print(f"Frame decoded successfully: shape={frame.shape}, dtype={frame.dtype}")
                    
                    # Verify modifications
                    cv2.circle(frame, (100, 100), 50, (0, 0, 255), -1)  # Add a red circle
                    cv2.putText(frame, "OpenCV Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    print("OpenCV modifications applied.")

                    # Display the frame
                    cv2.imshow("Video Stream", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting...")
                        break
                else:
                    print("Failed to decode frame, skipping...")
            else:
                print("No valid JPEG frame markers found, skipping chunk...")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera_process.terminate()
        camera_process.wait()

if __name__ == "__main__":
    main()

