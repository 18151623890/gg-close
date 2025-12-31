import os
import json

import torch
from torch.utils.data import Dataset
from PIL import Image


class ScreenshotDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # 图片目录：<root>/images
        self.img_dir = os.path.join(root, "images")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}. "
                "Expected structure: <root>/images/*.jpg|*.png"
            )

        all_files = sorted(os.listdir(self.img_dir))
        self.imgs = [
            x
            for x in all_files
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not self.imgs:
            raise RuntimeError(
                f"No image files found in {self.img_dir}. "
                "Supported extensions: .jpg, .jpeg, .png"
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 加载图片
        img_filename = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path).convert("RGB")

        # 对应的 JSON 标注：先找 images/xxx.json，再找 annotations/xxx.json
        file_base = os.path.splitext(img_filename)[0]
        json_path = os.path.join(self.img_dir, file_base + ".json")
        if not os.path.exists(json_path):
            json_path = os.path.join(self.root, "annotations", file_base + ".json")

        boxes = []
        labels = []

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            shapes = data.get("shapes", [])
            for shape in shapes:
                label_name = shape.get("label", "")
                points = shape.get("points", [])
                if not points:
                    # 没有点就跳过这个 shape（既不加 box 也不加 label）
                    continue

                # 当前是单类别任务：所有标注都映射为 1（背景为 0，模型内部处理）
                # 如果以后有多类别，可以在这里加一个 mapping，比如：
                # class_map = {"button": 1, "icon": 2}
                # labels.append(class_map.get(label_name, 1))
                labels.append(1)

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)

                boxes.append([xmin, ymin, xmax, ymax])

        # 转成 tensor
        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            # 没有框时，shape 应为 (0, 4)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        if labels:
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if boxes_tensor.numel() > 0:
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
                    boxes_tensor[:, 2] - boxes_tensor[:, 0]
            )
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        iscrowd = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
