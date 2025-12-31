import os
import random
import argparse

import torch
import torch.utils.data

from dataset import ScreenshotDataset
from model import get_model
import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_epoch_from_filename(path: str) -> int:
    """Try to infer epoch index from a filename like 'model_epoch_9.pth'."""
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    if not parts:
        return 0
    last = parts[-1]
    try:
        return int(last)
    except ValueError:
        return 0


def evaluate(model, data_loader, device):
    """
    Very simple evaluation: just run a forward pass on the test set
    to make sure everything works. No metrics are computed here.
    """
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            _ = model(images)  # predictions are ignored
    # caller (training loop) will set model.train() again as needed


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--resume",
        default="",
        help="Path to checkpoint or weights to resume training from (e.g. model_epoch_9.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train (from scratch or to continue from checkpoint)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    args = parser.parse_args()

    # --- Configuration ---
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 2  # 1 class (target) + background
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = 0.005
    random_seed = 42
    # ---------------------

    print(f"Using device: {device}")
    set_random_seeds(random_seed)

    # Dataset setup
    # Note: We assume running from project root, so path is 'data'
    dataset = ScreenshotDataset("data", get_transform(train=True))
    dataset_test_full = ScreenshotDataset("data", get_transform(train=False))

    print(f"Total images in dataset: {len(dataset)}")
    # 调试用：看第一张的标注
    img0, target0 = dataset[0]
    print("Sample target[0] boxes:", target0["boxes"])
    print("Sample target[0] labels:", target0["labels"])

    indices = torch.randperm(len(dataset)).tolist()
    test_split = max(1, int(len(dataset) * 0.2))
    dataset_train = torch.utils.data.Subset(dataset, indices[:-test_split])
    dataset_test = torch.utils.data.Subset(dataset_test_full, indices[-test_split:])

    if len(dataset_train) == 0:
        print("Error: No training data found. Please check the 'data' directory.")
        return

    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model setup
    model = get_model(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )

    # LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    # --- Resume from checkpoint if specified ---
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)

            # New style: full checkpoint dict
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "lr_scheduler" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                # Next epoch starts after the one stored in checkpoint
                start_epoch = int(checkpoint.get("epoch", 0)) + 1
                print(f"Loaded full checkpoint. Starting from epoch {start_epoch}.")
            else:
                # Old style: pure model state_dict
                model.load_state_dict(checkpoint)
                # Try to infer epoch from filename, e.g. 'model_epoch_9.pth'
                inferred_epoch = parse_epoch_from_filename(args.resume)
                start_epoch = inferred_epoch + 1
                print(
                    "Loaded model weights only (no optimizer/scheduler state). "
                    f"Continuing training from inferred epoch {start_epoch}."
                )
        else:
            print(f"Warning: Checkpoint {args.resume} not found! Starting from scratch.")

    # Training Loop
    # If resuming, we treat `num_epochs` as "additional epochs to run".
    end_epoch = start_epoch + num_epochs
    print(f"Training from epoch {start_epoch} to {end_epoch - 1} (inclusive).")

    for epoch in range(start_epoch, end_epoch):
        print(50 * "=")
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for iteration, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            epoch_loss += loss_value
            num_batches += 1

            if iteration % 10 == 0 or iteration == 1:
                print(f"Epoch: {epoch}, Iteration: {iteration}, Loss: {loss_value:.4f}")

        lr_scheduler.step()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("NaN")
        print(f"Epoch: {epoch} finished. Average Loss: {avg_loss:.4f}")

        # Simple evaluation pass (no metrics, just sanity check)
        print("Running evaluation on test set...")
        evaluate(model, data_loader_test, device)
        print("Evaluation pass completed.")

        checkpoint_dir = "checkpoints"  # 你想统一存放的目录名
        os.makedirs(checkpoint_dir, exist_ok=True)  # 没有就创建

        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print(50 * "=")
    print("Training Complete!")


if __name__ == "__main__":
    # 训练用：python src/train.py --resume checkpoints/model_epoch_9.pth
    main()
