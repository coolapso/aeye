# 🤖AEye 👀

A set of tools to train and run a **Convolutional Neural Network (CNN)** for object detection.

<p align="center">
  <img src="https://raw.githubusercontent.com/coolapso/aeye/refs/heads/main/images/demo.gif">
</p>

## Why This Project Exists

Have you ever wanted to create a custom AI model that analyzes video feeds or images and generates alerts or notifications—without breaking the bank or handing your data over to big corporations? This repository is designed to make that possible, offering an **accessible** and **cost-effective** way to deploy AI-powered image analysis.

When I moved to the **Arctic Circle**, where the **Northern Lights** are a common sight, I launched [The Arctic Skies](https://www.youtube.com/@TheArcticSkies) to **capture and share these breathtaking moments** with the world. While setting up my livestream, I had one thought:

> **"How cool would it be to have an AI model watching the sky and notifying me when an aurora appears?"**

After some research, I realized two things:

- I couldn't find any **pre-trained models** specifically for detecting the Northern Lights.
- There weren’t many **accessible tools** or **clear documentation** on how to build one.

So, I rolled up my sleeves and built this repository—a **set of tools to train and run AI models using TensorFlow and Keras**. And since I was already doing the work, I made it **generic and reusable**—so anyone can use it for their own projects.

Whether you're tracking auroras, wildlife, or any other visual phenomenon, this project is here to help you build your own **AI-powered image detection system**.

## Features

✅ **Image Sorting Assistance** – Sorting images manually can be tedious. Use `sort.py` to **quickly categorize images** by pressing `"y"` (yes) or `"n"` (no). The script automatically organizes them into the correct folder structure for training.

✅ **Efficient Image Resizing** – High-resolution images are **resource-intensive** and often unnecessary for training the sorting tool will **downscale images** to an optimal resolution.

✅ **Train Your Own AI Model** – Use `train.py` to **train a custom TensorFlow/Keras model** on your dataset. Currently, only **binary classification** (detected/not detected) is supported. If you need multi-class detection, feel free to open an issue—I’d be happy to add support for it!

✅ **Real-Time Model Execution with Metrics** – The main application **runs your trained model** and generates **Prometheus-formatted metrics**, which can be scraped by **Prometheus** or other monitoring tools.

✅ **Dockerized Deployment** – Easily deploy your model in a **Docker container** using the provided image, or build your own from the base image.

## Requirements

- **Python 3.12**
- **OpenCV & TensorFlow**
- Check your OS documentation on how to install them.

---

## How It Works

Even though this repository simplifies the process significantly, there’s still some work to do! Hopefully, this documentation will guide you through it smoothly.

### 1️⃣ Collect Training Data

To train the AI model, you need an **initial dataset**. While smaller datasets work, I recommend:
- **At least 800 images** containing the object you want to detect.
- **At least 800 images** that **do not** contain the object.
- (Optional) Another **separate dataset for validation**, not used during training.

For **best performance**, images should be resized to a smaller resolution. Training on full-resolution images provides **little benefit** but requires significantly more resources.

Sorting images can be tedious, but the `sort.py` script helps automate this process.

---

## 2️⃣ Set Up the Environment

Before using the tools, set up a **virtual environment** and install dependencies:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r tools/requirements.txt
```

---

## 3️⃣ Prepare the Images
Use `sort.py` to categorize images:

<p align="center">
  <img src="https://raw.githubusercontent.com/coolapso/aeye/refs/heads/main/images/sort-demo.gif">
</p>


```sh
python tools/sort.py -r dataset -f allimages
```

### Usage:
```sh
usage: sort.py [-h] [-r DATASET_ROOT] [-f FRAMES_DIR]

Tool to assist with image sorting before using them for training.

options:
  -h, --help            Show this help message and exit
  -r DATASET_ROOT, --root DATASET_ROOT
                        The root directory of the dataset
  -f FRAMES_DIR, --frames FRAMES_DIR
                        Directory containing images to review
```

**How it works:**
- Press **"y"** for images **containing** the object.
- Press **"n"** for images **without** the object.

The script will automatically:

✅ Move files to `00-detected/` and `01-not-detected/` directories.

✅ Store **both original and resized images** for training.

> [!IMPORTANT]
> The **numbers in directory names are important**—they define the dataset structure used for training and metric generation!

---

## 4️⃣ Train the Model
Use `train.py` to train the AI model:

<p align="center">
  <img src="https://raw.githubusercontent.com/coolapso/aeye/refs/heads/main/images/train-demo.gif">
</p>

```sh
python tools/train.py --dataset dataset/ --validation validation-dataset/ --output models/my_model.keras
```

---

## 5️⃣ Run the Model

The application loads the trained model and generates **Prometheus metrics** for monitoring.

### Configuration via Environment Variables:

| Variable Name            | Description | Default Value |
|--------------------------|-------------|--------------|
| `VIDEO_SOURCE`           | Video file or RTSP stream for analysis | - |
| `MODEL_NAME`             | Model filename (must be in `models/` directory) | - |
| `MODE`                   | Operation mode (`default`, `testing`, `classify`, `dynamic`) | `default` |
| `FRAMES_OUTPUT_ROOT_DIR` | Where images are stored in `classify` or `dynamic` mode | `dataset/` |
| `READ_ALL_FRAMES`        | Analyze every frame instead of 1 FPS | `false` |

Metrics are available at [`http://localhost:8000`](http://localhost:8000).

### Running Locally

Activate the virtual environment and run:

```sh
VIDEO_SOURCE="..." MODEL_NAME="my_model.keras" python src/aeye.py
```

---

### Running with Docker

```sh
docker run -tid -p 8000:8000 -v models/:models/ \
  -e VIDEO_SOURCE="..." -e MODEL_NAME="my_model.keras" my_docker_image
```

---

#### 🛠️ Build Your Own Docker Image

While the main Docker image **contains models trained for** [The Arctic Skies](https://youtube.com/@TheArcticSkies), a **base image** without any pre-bundled models is also provided. You can use it to build your own Docker image and bundle your custom model.

**Example Dockerfile:**

```dockerfile
FROM ghcr.io/coolapso/AICoach:latest-base
COPY models/yourmodel.keras models/yourmodel.keras
```

You can also set environment variables directly in the Dockerfile instead of passing them at runtime:

```dockerfile
FROM ghcr.io/coolapso/AICoach:latest-base
COPY models/yourmodel.keras models/yourmodel.keras
ENV MODEL_NAME=yourmodel.keras
ENV VIDEO_SOURCE="..."
```

## Exposed metrics

| Metric name     | Type  | Description                                                           |
|-----------------|-------|-----------------------------------------------------------------------|
| aeye_detected   | Gauge | If object is detected or not, 0 not detected, 1 detected, 2 uncertain |
| aeye_confidence | Gauge | The confidence of the detection                                       |

# 🤝 Contributions

All contributions are welcome!

I’m still learning in this domain, and I’d love to make this project more **user-friendly and accessible** for a broader audience—not just makers and developers!

If you’d like to support the project:
[:heart: Sponsor Me](https://github.com/sponsors/coolapso) or

[![Buy Me a Coffee](https://cdn.buymeacoffee.com/buttons/default-yellow.png)](https://www.buymeacoffee.com/coolapso)

