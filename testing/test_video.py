import cv2
import numpy as np
import subprocess

def main():
    # Start rpicam-vid as a subprocess
    camera_process = subprocess.Popen(
        ["rpicam-vid", "--width", "640", "--height", "480", "--timeout", "0", "--output", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    # MJPEG buffer
    mjpeg_buffer = b''

    try:
        while True:
            # Read from the subprocess stdout
            chunk = camera_process.stdout.read(4096)
            if not chunk:
                print("Camera process stopped sending data.")
                break

            # Append chunk to buffer
            mjpeg_buffer += chunk

            # Look for JPEG frame markers
            start_idx = mjpeg_buffer.find(b'\xff\xd8')  # JPEG start marker
            end_idx = mjpeg_buffer.find(b'\xff\xd9')  # JPEG end marker

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # Extract the complete JPEG frame
                frame_data = mjpeg_buffer[start_idx:end_idx + 2]
                mjpeg_buffer = mjpeg_buffer[end_idx + 2:]  # Remove processed frame from buffer

                # Decode the frame
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # Display the frame
                    cv2.imshow("MJPEG Stream", frame)
                else:
                    print("Failed to decode frame.")

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera_process.terminate()
        camera_process.wait()

if __name__ == "__main__":
    main()

