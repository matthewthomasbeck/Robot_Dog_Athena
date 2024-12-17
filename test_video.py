import sys
import cv2
import numpy as np

def main():
    width = 2304
    height = 1296

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

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

