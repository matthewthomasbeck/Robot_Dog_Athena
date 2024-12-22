########## IMPORT DEPENDENCIES ##########

from openvino.runtime import Core
import numpy as np
import cv2
import subprocess


########## FUNCTION DEFINITIONS ##########

def load_and_compile_model(model_xml_path, device_name="MYRIAD"):
    """
    Loads and compiles an OpenVINO model.

    :param model_xml_path: Path to the model XML file.
    :param device_name: Device to compile the model on (e.g., CPU, MYRIAD).
    :return: (compiled_model, input_layer, output_layer)
    """
    ie = Core()
    try:
        # The BIN path is derived by replacing .xml with .bin if needed:
        # If you want to keep them separate, you can parametrize model_bin_path as well.
        model_bin_path = model_xml_path.replace(".xml", ".bin")

        model = ie.read_model(model=model_xml_path)
        compiled_model = ie.compile_model(model=model, device_name=device_name)
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)

        print(f"Model loaded and compiled successfully on {device_name}.")
        print(f"Model input shape: {input_layer.shape}")

        return compiled_model, input_layer, output_layer
    except Exception as e:
        print(f"Failed to load and compile model: {e}")
        exit(1)


def test_with_dummy_input(compiled_model, input_layer, output_layer):
    """
    Sends dummy input to the compiled model to ensure it works.

    :param compiled_model: The compiled OpenVINO model.
    :param input_layer: The model's input layer.
    :param output_layer: The model's output layer.
    """
    try:
        dummy_input_shape = input_layer.shape
        dummy_input = np.ones(dummy_input_shape, dtype=np.float32)
        results = compiled_model([dummy_input])[output_layer]
        print("Dummy input test passed!")
    except Exception as e:
        print(f"Error with dummy input: {e}")
        exit(1)


def start_camera_process(width=640, height=480, framerate=30):
    """
    Starts the rpicam-vid process to output MJPEG data via stdout.

    :param width: Desired camera width.
    :param height: Desired camera height.
    :param framerate: Desired framerate.
    :return: Subprocess object.
    """
    camera_process = subprocess.Popen(
        [
            "rpicam-vid", "--width", str(width), "--height", str(height),
            "--framerate", str(framerate), "--timeout", "0",
            "--output", "-", "--codec", "mjpeg", "--nopreview"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    return camera_process


def inference_loop(compiled_model, input_layer, output_layer, camera_process):
    """
    Continuously reads frames from the camera process, performs inference,
    and displays the resulting frames.

    :param compiled_model: The compiled OpenVINO model.
    :param input_layer: The model's input layer.
    :param output_layer: The model's output layer.
    :param camera_process: The subprocess object capturing the camera feed.
    """
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
                    try:
                        # Convert frame to RGB if required by the model
                        # (some models expect BGR, some expect RGB; adjust as needed)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Resize frame to the model's expected input size (e.g., 300x300 or 256x256)
                        input_blob = cv2.resize(frame_rgb, (256, 256))

                        # Transpose to match the model's expected layout: (C, H, W)
                        input_blob = input_blob.transpose(2, 0, 1)

                        # Add batch dimension
                        input_blob = np.expand_dims(input_blob, axis=0)

                        # Ensure the data type is correct
                        input_blob = input_blob.astype(np.float32)

                        # Perform inference
                        results = compiled_model([input_blob])[output_layer]

                        # Draw detections on the frame
                        for detection in results[0][0]:
                            confidence = detection[2]
                            if confidence > 0.5:  # Confidence threshold
                                xmin, ymin, xmax, ymax = map(
                                    int,
                                    detection[3:7] * [
                                        frame.shape[1],
                                        frame.shape[0],
                                        frame.shape[1],
                                        frame.shape[0]
                                    ]
                                )
                                label = f"ID {int(detection[1])}: {confidence:.2f}"
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    label,
                                    (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2
                                )

                        # Display the frame
                        cv2.imshow("OpenVINO Inference", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    except Exception as inference_error:
                        print(f"Error during inference: {inference_error}")
                else:
                    print("Failed to decode frame, skipping...")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera_process.terminate()
        camera_process.wait()


########## MAIN EXECUTION ##########
if __name__ == "__main__":
    # Adjust paths as needed
    MODEL_XML = "/home/matthewthomasbeck/Projects/Robot_Dog/model/person-detection-0200.xml"
    DEVICE_NAME = "MYRIAD"

    # 1. Load and compile the model
    compiled_model, input_layer, output_layer = load_and_compile_model(MODEL_XML, DEVICE_NAME)

    # 2. Test with dummy input
    test_with_dummy_input(compiled_model, input_layer, output_layer)

    # 3. Start the camera
    camera_process = start_camera_process(width=640, height=480, framerate=30)

    # 4. Run inference loop
    inference_loop(compiled_model, input_layer, output_layer, camera_process)
