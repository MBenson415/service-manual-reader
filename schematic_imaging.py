import math
import os

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def preprocessing_enabled() -> bool:
    return os.environ.get("SCHEMATIC_PREPROCESS", "1").lower() not in {"0", "false", "no"}


def estimate_skew(image: Image.Image) -> float:
    preview = image.convert("L")
    preview.thumbnail((1600, 1600))
    pixels = np.asarray(ImageOps.autocontrast(preview))
    edges = cv2.Canny(pixels, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=40,
        minLineLength=max(40, preview.width // 8), maxLineGap=10,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines.reshape(-1, 4):
        left, top, right, bottom = map(int, line)
        angle = math.degrees(math.atan2(bottom - top, right - left))
        if abs(angle) <= 3:
            angles.append(angle)
    if len(angles) < 5:
        return 0.0
    median = float(np.median(angles))
    if float(np.median(np.abs(np.asarray(angles) - median))) > 0.4:
        return 0.0
    return median if abs(median) >= 0.15 else 0.0


def preprocess_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    if not preprocessing_enabled():
        return gray
    angle = estimate_skew(gray)
    if angle:
        gray = gray.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    pixels = np.asarray(ImageOps.autocontrast(gray, cutoff=0))
    enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(pixels)
    return Image.fromarray(enhanced).filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))