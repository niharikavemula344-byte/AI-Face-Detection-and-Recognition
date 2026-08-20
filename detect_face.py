"""Detect faces in an image with OpenCV's Haar cascade classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def detect_faces(image_path: Path, cascade_path: Path, output_path: Path) -> int:
    """Detect faces, save an annotated image, and return the face count."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise ValueError(f"Could not load cascade: {cascade_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (46, 204, 113), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write output image: {output_path}")
    return len(faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect faces in a static image.")
    parser.add_argument("image", nargs="?", type=Path, default=Path("test.jpg"))
    parser.add_argument("--cascade", type=Path, default=Path("haarcascade_frontalface_default.xml"))
    parser.add_argument("--output", type=Path, default=Path("output/detected-faces.jpg"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = detect_faces(args.image, args.cascade, args.output)
    print(f"Detected {count} face(s). Saved result to {args.output}.")
