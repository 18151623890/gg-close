import argparse
import os
import time
from functools import wraps

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw  # ImageFont 可按需导入

from model import get_model


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 更精确的时间测量
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} 执行耗时: {elapsed_time:.6f} 秒")
        return result

    return wrapper


def load_model_for_inference(model_path, num_classes=2):
    """加载训练好的模型，兼容纯 state_dict 和完整 checkpoint 两种格式。"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(num_classes)

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None, device

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 兼容两种保存方式：
    # 1) torch.save(model.state_dict(), path)
    # 2) torch.save({'model': model.state_dict(), ...}, path)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


@timing_decorator
def get_prediction(img_path, threshold, model, device):
    if model is None:
        return None, None, None

    img = Image.open(img_path).convert("RGB")

    # 简单的推理变换：只转 tensor，保持原始尺寸
    transform = T.ToTensor()
    img_tensor = transform(img).to(device)  # [C, H, W]

    with torch.no_grad():
        # detection 模型期望输入是 List[Tensor]
        prediction = model([img_tensor])

    # prediction 是长度为 1 的列表，每个元素是一个 dict
    output = prediction[0]
    pred_boxes = output["boxes"].cpu()
    pred_scores = output["scores"].cpu()
    pred_labels = output["labels"].cpu()

    # 按阈值过滤
    keep = pred_scores > threshold
    return pred_boxes[keep], pred_scores[keep], img


def draw_boxes(img, boxes, scores):
    draw = ImageDraw.Draw(img)
    # 如需自定义字体，可使用：
    # font = ImageFont.truetype("arial.ttf", 15)

    for i, box in enumerate(boxes):
        score = scores[i].item()
        box = box.tolist()  # [xmin, ymin, xmax, ymax]

        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{score:.2f}", fill="red")
        # 如需字体：draw.text((box[0], box[1]), f"{score:.2f}", fill="red", font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--model", "-m", required=True, help="Path to model weights/checkpoint")
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold")

    args = parser.parse_args()

    model, device = load_model_for_inference(args.model, num_classes=2)
    print("loading model done")
    boxes, scores, img = get_prediction(args.image, args.threshold, model, device)

    if boxes is not None:
        result_img = draw_boxes(img, boxes, scores)
        result_img.show()

        if not os.path.exists("results"):
            os.makedirs("results")
        save_path = os.path.join("results", "prediction.jpg")
        result_img.save(save_path)
        print(f"Result saved to {save_path}")
    else:
        print("No prediction made.")


if __name__ == "__main__":
    main()
