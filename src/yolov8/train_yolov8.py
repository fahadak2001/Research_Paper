from ultralytics import YOLO
import os
import torch

def check_environment():
    print("\n===== SYSTEM CHECK =====")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU. Training will be slower.")
    print("========================\n")


def main():
    # Check environment first
    check_environment()

    # Path to data.yaml
    data_yaml_path = "data/data.yaml"

    # Load a pretrained YOLOv8 model (choose size: n, s, m, l, x)
    model = YOLO("yolov8s.pt")

    # Train the model
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs_train",
        name="vehicle_detector",
        exist_ok=True
    )

    # Validate model
    print("\n===== VALIDATING MODEL =====")
    metrics = model.val()
    print("Validation Results:", metrics)

    # Run inference on some test images automatically
    test_images_dir = "data/test/images"
    if os.path.exists(test_images_dir):
        print("\n===== RUNNING TEST INFERENCE =====")
        model.predict(
            source=test_images_dir,
            save=True,
            project="runs_inference",
            name="test_results",
            exist_ok=True
        )
        print("Predictions saved in runs_inference/test_results/")
    else:
        print("\nNo test folder found, skipping inference.")

    print("\nTraining complete! Check the following folders:")
    print(" - runs_train/vehicle_detector/weights/best.pt")
    print(" - runs_train/vehicle_detector/weights/last.pt")
    print(" - runs_inference/test_results/ (if test images existed)")


if __name__ == "__main__":
    main()
