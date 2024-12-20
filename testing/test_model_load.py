from openvino.runtime import Core

def main():
    model_path = "/home/matthewthomasbeck/Projects/Robot_Dog/model/yolov8n.onnx"
    ie = Core()
    try:
        model = ie.read_model(model=model_path)
        print("ONNX model loaded successfully!")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")

if __name__ == "__main__":
    main()

