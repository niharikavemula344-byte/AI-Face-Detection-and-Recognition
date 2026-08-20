# Face Detection with OpenCV

I built this project to understand the basics of face detection with OpenCV. It takes an image, finds frontal faces with a Haar cascade, draws boxes around them, and saves the result.

> This project performs **face detection**, not identity recognition. It does not identify who a person is.

## What it does

- Validates image and classifier inputs
- Detects multiple frontal faces
- Saves results without requiring a desktop GUI
- Accepts custom input, classifier, and output paths
- Runs entirely offline

## Tech stack

- Python 3.9+
- OpenCV
- Haar cascade classification

## Quick start

```bash
git clone https://github.com/niharikavemula344-byte/AI-Face-Detection-and-Recognition.git
cd AI-Face-Detection-and-Recognition
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python detect_face.py test.jpg
```

The annotated image is written to `output/detected-faces.jpg` by default.

## Usage

```bash
python detect_face.py path/to/image.jpg --output output/result.jpg
python detect_face.py path/to/image.jpg --cascade path/to/cascade.xml
```

## Project structure

```text
.
├── detect_face.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── test.jpg
```

## What I learned

This project helped me understand grayscale preprocessing, cascade classifiers, command-line arguments, and input validation.

## Limitations

Haar cascades work best on well-lit, front-facing faces. For production use, a modern deep-learning detector and explicit consent/privacy controls are recommended.

## License

This repository does not currently include a license. All rights are reserved by the author.
