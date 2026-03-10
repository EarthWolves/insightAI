"""
Grounding DINO — open-set object detection with custom text prompts.

Usage:
    python detect.py --image path/to/image.jpg --classes "cat . dog . person"
    python detect.py --image path/to/image.jpg --classes "cat . dog" --threshold 0.35 --save
"""

import argparse
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # swap for -base for higher accuracy


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_id: str = MODEL_ID):
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    return processor, model, device


def format_classes(classes: str) -> str:
    """
    Grounding DINO expects classes separated by ' . ' with a trailing ' .'
    Input can be comma-separated ('cat, dog') or already formatted ('cat . dog').
    """
    if " . " in classes:
        text = classes.strip()
    else:
        parts = [c.strip().lower() for c in classes.split(",") if c.strip()]
        text = " . ".join(parts)
    if not text.endswith("."):
        text = text + " ."
    return text


def detect(
    image_path: str,
    classes: str,
    threshold: float = 0.3,
    processor=None,
    model=None,
    device=None,
) -> list[dict]:
    """
    Run Grounding DINO on an image.

    Returns a list of dicts:
        {"label": str, "score": float, "box": [x1, y1, x2, y2]}
    """
    image = Image.open(image_path).convert("RGB")
    text_prompt = format_classes(classes)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["text_labels"], results["boxes"]
    ):
        detections.append(
            {
                "label": label,
                "score": float(score),
                "box": [round(v, 2) for v in box.tolist()],
            }
        )

    return detections, image


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    colors = [
        "#FF3B30", "#34C759", "#007AFF", "#FF9500",
        "#AF52DE", "#FF2D55", "#5AC8FA", "#FFCC00",
    ]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{det['label']} {det['score']:.2f}"
        bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label_text, fill="white", font=font)

    return image


def main():
    parser = argparse.ArgumentParser(description="Grounding DINO custom class detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--classes",
        required=True,
        help='Classes to detect, comma-separated: "cat, dog, person"',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated image alongside the input",
    )
    args = parser.parse_args()

    processor, model, device = load_model(args.model)
    detections, image = detect(
        args.image, args.classes, args.threshold, processor, model, device
    )

    if not detections:
        print("No detections above threshold.")
        return

    print(f"\nDetected {len(detections)} object(s):\n")
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        print(f"  {d['label']:<20} score={d['score']:.3f}  box=[{x1}, {y1}, {x2}, {y2}]")

    if args.save:
        annotated = draw_detections(image.copy(), detections)
        out_path = args.image.rsplit(".", 1)[0] + "_detected.jpg"
        annotated.save(out_path)
        print(f"\nSaved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
