import os
import yaml
import argparse
from ultralytics import YOLO


def setup_dataset(data_path):
    """Simple check of dataset configuration"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data configuration file does not exist: {data_path}")

    # Read data configuration
    with open(data_path, 'r') as file:
        data_config = yaml.safe_load(file)

    print(f"Dataset configuration:")
    print(f"- Number of classes: {data_config.get('nc', 'N/A')}")
    print(f"- Class names: {data_config.get('names', 'N/A')}")

    return data_config


def train_license_plate_model(data_path, model_name, epochs, img_size, batch_size, output_dir, device):
    """Train license plate recognition model and save it"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using model: {model_name}")
    print(f"Starting license plate recognition model training...")
    print(f"- Dataset: {data_path}")
    print(f"- Device: {device}")
    print(f"- Batch size: {batch_size}")
    print(f"- Image size: {img_size}")
    print(f"- Training epochs: {epochs}")

    try:
        # Load model
        model = YOLO(model_name)

        # Start training
        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=output_dir,
            name="license_plate_detection",
            exist_ok=True,

            # Add early stopping configuration
            patience=args.patience,

            # Add loss weights
            box=args.box,
            cls=args.cls,
            dfl=args.dfl,

            # Add data augmentation parameters
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,

            # Add learning rate parameters
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,

            save=True,
            verbose=True,
            amp=False,
            cos_lr=True
        )

        # Get best model path
        best_model_path = os.path.join(output_dir, "license_plate_detection", "weights", "best.pt")

        if os.path.exists(best_model_path):
            print(f"✅ Training successful! Model saved to: {best_model_path}")
            return best_model_path
        else:
            print(f"❌ Training may have completed, but model file not found: {best_model_path}")
            last_model_path = os.path.join(output_dir, "license_plate_detection", "weights", "last.pt")
            if os.path.exists(last_model_path):
                print(f"✅ But found the last checkpoint: {last_model_path}")
                return last_model_path
            return None

    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        return None


if __name__ == "__main__":
    # Set environment variables to avoid OpenMP errors
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(description="Train license plate recognition model")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML file path")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default="runs/train", help="Output directory")
    parser.add_argument("--device", type=str, default="0", help="Training device: 0, 0,1, cpu")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience, set to 0 to disable early stopping")

    # Loss weight parameters
    parser.add_argument("--box", type=float, default=7.5, help="Bounding box loss weight")
    parser.add_argument("--cls", type=float, default=0.5, help="Class loss weight")
    parser.add_argument("--dfl", type=float, default=1.5, help="Distribution focal loss weight")

    # Data augmentation parameters
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation angle range")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation range")
    parser.add_argument("--scale", type=float, default=0.5, help="Scaling range")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear transformation range")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective transformation range")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation probability")

    # Add learning rate related parameters
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs")

    args = parser.parse_args()

    # Simple check of dataset
    setup_dataset(args.data)

    # Train model
    model_path = train_license_plate_model(
        args.data,
        args.model,
        args.epochs,
        args.img_size,
        args.batch,
        args.output,
        args.device
    )

    if model_path:
        print(f"\nTraining completed! You can use the model at the following path for license plate recognition tasks:\n{model_path}")
    else:
        print("\nTraining was not successfully completed.")