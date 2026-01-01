import os
import random
import argparse
import torch
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置每个进程的随机种子（确保数据shuffle的一致性）
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)
    random.seed(42 + rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def train(rank, world_size, args):
    """每个进程的训练函数"""
    # 设置分布式环境
    setup(rank, world_size)

    # --- Configuration ---
    device = torch.device(f"cuda:{rank}")
    num_classes = 2  # 1 class (target) + background
    num_epochs = args.epochs
    batch_size = args.batch_size
    # 0.005 × (gpu个数 × 原来batch_size / 原来batch_size)
    # 0.005 × (12 / 4) = 0.005 × 3 = 0.015
    learning_rate = 0.015
    # ---------------------

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Using device: {device}")

    # Dataset setup - 每个进程使用不同的数据子集
    dataset = ScreenshotDataset("data", get_transform(train=True))
    dataset_test_full = ScreenshotDataset("data", get_transform(train=False))

    if rank == 0:
        print(f"Total images in dataset: {len(dataset)}")
        # 调试用：看第一张的标注
        img0, target0 = dataset[0]
        print("Sample target[0] boxes:", target0["boxes"])
        print("Sample target[0] labels:", target0["labels"])

    # 创建分布式sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # 测试集不需要分布式sampler（只在rank 0上评估）
    indices = torch.randperm(len(dataset)).tolist()
    test_split = max(1, int(len(dataset) * 0.2))
    dataset_test = torch.utils.data.Subset(dataset_test_full, indices[-test_split:])

    if len(dataset) == 0:
        if rank == 0:
            print("Error: No training data found. Please check the 'data' directory.")
        cleanup()
        return

    # 注意：每个进程的batch_size是args.batch_size，总batch_size = args.batch_size * world_size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,  # 可以增加workers以提高数据加载速度
        collate_fn=collate_fn,
        pin_memory=True,  # 加速数据传输到GPU
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Model setup
    model = get_model(num_classes)
    model.to(device)

    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=0.0003
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
            if rank == 0:
                print(f"Resuming training from {args.resume}...")

            # 所有进程加载相同的checkpoint
            checkpoint = torch.load(args.resume, map_location=device)

            # 处理DDP模型的state_dict（去掉'module.'前缀）
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
                # 如果checkpoint是DDP保存的，需要处理前缀
                if any(key.startswith('module.') for key in state_dict.keys()):
                    # 去掉'module.'前缀
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # 加载模型参数
                model.module.load_state_dict(state_dict)

                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "lr_scheduler" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

                start_epoch = int(checkpoint.get("epoch", 0)) + 1
                if rank == 0:
                    print(f"Loaded full checkpoint. Starting from epoch {start_epoch}.")
            else:
                # Old style: pure model state_dict
                state_dict = checkpoint
                if any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.module.load_state_dict(state_dict)

                inferred_epoch = parse_epoch_from_filename(args.resume)
                start_epoch = inferred_epoch + 1
                if rank == 0:
                    print(
                        "Loaded model weights only (no optimizer/scheduler state). "
                        f"Continuing training from inferred epoch {start_epoch}."
                    )
        else:
            if rank == 0:
                print(f"Warning: Checkpoint {args.resume} not found! Starting from scratch.")

    # Training Loop
    end_epoch = start_epoch + num_epochs
    if rank == 0:
        print(f"Training from epoch {start_epoch} to {end_epoch - 1} (inclusive).")

    for epoch in range(start_epoch, end_epoch):
        # 设置epoch给sampler（确保每个epoch的shuffle不同）
        train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for iteration, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            epoch_loss += loss_value
            num_batches += 1

            if rank == 0 and (iteration % 10 == 0 or iteration == 1):
                print(f"Epoch: {epoch}, Iteration: {iteration}, Loss: {loss_value:.4f}")

        lr_scheduler.step()

        # 同步所有进程的损失
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("NaN")
        if rank == 0:
            print(f"Epoch: {epoch} finished. Average Loss: {avg_loss:.4f}")

            # 只在rank 0上评估和保存checkpoint
            print("Running evaluation on test set...")
            evaluate(model, data_loader_test, device)
            print("Evaluation pass completed.")

            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存时使用model.module.state_dict()去掉DDP包装
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    if rank == 0:
        print(50 * "=")
        print("Training Complete!")

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training with DDP")
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
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="Number of GPUs per node (use -1 for all available)",
    )
    args = parser.parse_args()

    # 计算world_size
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()

    world_size = args.gpus * args.nodes

    if world_size > 1:
        # 使用多进程启动DDP
        mp.spawn(
            train,
            args=(world_size, args),
            nprocs=args.gpus,
            join=True
        )
    else:
        # 单GPU训练
        train(0, 1, args)


if __name__ == "__main__":
    # 新的训练命令：
    # CUDA_VISIBLE_DEVICES=1,2,3 python src/train_gpu.py --batch_size 4
    main()
