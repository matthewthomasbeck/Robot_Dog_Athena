import cv2
import numpy as np
#from openvino.inference_engine import IECore

# Initialize the IECore
#ie = IECore()
#
## Load the network
#net = ie.read_network(model="path_to_model.xml", weights="path_to_model.bin")
#exec_net = ie.load_network(network=net, device_name="CPU")
#
## Get the input and output layer names
#input_blob = next(iter(net.input_info))
#output_blob = next(iter(net.outputs))

# Initialize the camera
#camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

#pipeline = "libcamerasrc ! video/x-raw,width=4608,height=2592,framerate=14/1 ! videoconvert ! appsink"
#print(cv2.getBuildInformation())
# Initialize the camera with GStreamer pipeline
#camera = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
if not camera.isOpened():
    print("Error: Could not open video device")
    exit()
#4608x2592
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2304)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1296)
camera.set(cv2.CAP_PROP_FPS, 56.03)
#2304x1296
while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        print(ret)
        break
#
#    # Pre-process the frame
#    input_image = cv2.resize(frame, (net.input_info[input_blob].input_data.shape[2], net.input_info[input_blob].input_data.shape[3]))
#    input_image = input_image.transpose((2, 0, 1))
#    input_image = input_image.reshape(1, 3, net.input_info[input_blob].input_data.shape[2], net.input_info[input_blob].input_data.shape[3])
#
#    # Run inference
#    res = exec_net.infer(inputs={input_blob: input_image})
#
#    # Process the output
#    output = res[output_blob]

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#
## Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
