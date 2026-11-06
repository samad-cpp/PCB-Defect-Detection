# AI-Based PCB Defect Detection

An AI-assisted Printed Circuit Board (PCB) defect detection application built with **Python, Ultralytics YOLO, OpenCV, and Streamlit**. The application supports image-based inspection and live camera inspection through a simple web interface.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Defect Classes](#defect-classes)
- [Models](#models)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [Camera Configuration](#camera-configuration)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Project Overview

This project uses trained YOLO object-detection models to identify potential defects in PCB images. A Streamlit dashboard provides two inspection methods:

1. **Image upload:** Inspect a PCB image using full-image inference or overlapping-tile inference.
2. **Live camera:** Inspect a selected region of a live camera feed using a Region of Interest (ROI).

The application displays detected bounding boxes, class labels, confidence scores, and a detection summary.

## Features

- PCB inspection using two trained YOLO models.
- Upload images in JPG, JPEG, PNG, BMP, and WEBP formats.
- Full-image inference.
- Overlapping-tile inference to inspect large images in smaller regions.
- Live inspection using a laptop camera, USB webcam, or DroidCam.
- Automatic discovery of supported physical cameras.
- Adjustable ROI for live camera inspection.
- Manual or continuous inference on the selected ROI.
- Annotated detection results and downloadable output images.
- Detection counts and mean confidence summary.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Ultralytics YOLO | Object detection |
| PyTorch | Deep learning framework |
| Streamlit | Web application interface |
| OpenCV | Image processing and camera access |
| NumPy | Numerical operations |
| Pillow | Image loading and handling |
| cv2-enumerate-cameras | Physical camera discovery |

## Defect Classes

The PCB Defect Model is configured for these six classes:

1. Mouse bite
2. Spur
3. Missing hole
4. Short
5. Open circuit
6. Spurious copper

**Note:** The DeepPCB model uses a different class mapping. Select the correct model and verify its class labels before interpreting predictions.

## Datasets

### 1. PCB Defect Dataset
- Source: [Kaggle - PCB Defect Dataset](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset)
- Uploaded by: Norbert Elter
- Contains 10,668 images with YOLO annotations.
- Includes six PCB defect classes.
- Dataset is based on the PKU-Market-PCB research dataset.

### 2. DeepPCB Dataset
- Source: [GitHub - DeepPCB](https://github.com/tangsanli5201/DeepPCB)
- Contains paired PCB template and test images.
- Includes six types of PCB defects.

Please refer to the original dataset sources for their usage
and redistribution conditions.



## Models

The application includes two trained model weights:

| Model | File |
|---|---|
| PCB Defect Model | `models/pcb_yolo_baseline_w0.pt` |
| DeepPCB Model | `models/deeppcb_yolo_baseline.pt` |

Both model files must be present in the `models/` directory for the corresponding options to work.

## Model Evaluation

The following metrics were recorded for the PCB baseline model:

| Metric | Score |
|---|---:|
| Precision | 0.980 |
| Recall | 0.983 |
| mAP@50 | 0.989 |
| mAP@50-95 | 0.559 |

- **Precision:** Indicates how many predicted defects were correct.
- **Recall:** Indicates how many actual labelled defects were detected.
- **mAP@50:** Mean average precision at an IoU threshold of 0.50.
- **mAP@50-95:** Mean average precision averaged across IoU thresholds from 0.50 to 0.95.

These are evaluation metrics, not overall classification accuracy. Confirm the model and evaluation run associated with these values before publishing them as final results.

## Project Structure

```text
PCB-Defect-Detection/
│
├── app.py
├── requirements.txt
├── requirements-gpu.txt
├── models/
│   ├── pcb_yolo_baseline_w0.pt
│   └── deeppcb_yolo_baseline.pt
│
├── notebooks/
├── src/
├── results/                 # Local training outputs; may not be included in GitHub
├── datasets/                # Local dataset files; may not be included in GitHub
├── .gitignore
└── README.md
```

The folders shown for notebooks, source code, results, and datasets should be adjusted to match the files actually committed to the repository. Model weights are expected in `models/`.

## Installation and Setup

### Prerequisites

- Python installed on your computer. Python 3.12 is the intended version for the setup instructions; dependency installation should be tested in a clean Python 3.12 environment before release.
- Git, if you want to clone the repository using the command line.
- An NVIDIA GPU is optional. The application can also be run without one, although inference may be slower.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PCB-Defect-Detection
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### 2. Install dependencies

**Without a virtual environment**

A virtual environment is optional. You can install the dependencies directly into your existing Python environment:

```bash
python -m pip install -r requirements.txt
```

**Using a virtual environment (recommended)**

A virtual environment keeps this project's packages separate from other Python projects and helps prevent dependency conflicts.

Create one:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. GPU installation (optional)

For the NVIDIA CUDA 12.8 dependency configuration provided with this project, install:

```bash
python -m pip install -r requirements-gpu.txt
```

Use this instead of the standard requirements installation when setting up the configured GPU environment. Confirm that your NVIDIA driver and GPU are compatible with the installed PyTorch build.

## Running the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that address in your web browser to use the dashboard.

## Using the Application

### Image Upload

1. Open the **Upload image** tab.
2. Select the trained model from the sidebar.
3. Choose **Full image** or **Overlapping tiles** inference.
4. Upload a PCB image.
5. Select **Run inspection**.
6. Review the annotated image, predicted classes, confidence scores, and detection summary.
7. Download the annotated result if required.

### Live Camera Inspection

1. Open the **Live camera** tab.
2. Select **Laptop Camera**, **USB Camera**, or **DroidCam**.
3. Select a detected physical camera when using a laptop or USB camera.
4. For DroidCam, enter the phone's current IP address and port.
5. Select **Start camera**.
6. Adjust the ROI rectangle to select the region to inspect.
7. Enable continuous inference or select **Analyze current boxed area** for a single inspection.

Live camera inference processes only the selected ROI. It does not use overlapping tiles or segmentation.

## Camera Configuration

### Laptop and USB Cameras

The application discovers physical cameras at runtime using `cv2-enumerate-cameras`. Connect the camera and use **Refresh cameras** if it does not appear in the list.

Windows may not always identify an internal webcam consistently. Depending on the device, a built-in camera may appear under the USB camera category.

### DroidCam

The application connects directly to the DroidCam video stream over the local network.

- Keep the DroidCam app open and streaming on your phone.
- Make sure the phone and computer can communicate over the same network.
- Enter the IP address and port displayed by DroidCam in the application.
- The phone's IP address may change when it reconnects to the network.

The default IP and port shown in the application are examples from the development setup. Replace them with the values shown on your phone.

## Limitations

- Detection quality depends on image quality, lighting, PCB appearance, and the training data.
- The two models may use different class mappings. Verify the selected model before interpreting labels.
- Live camera inspection requires a working camera connection and is intended for local execution.
- Camera access from a hosted Streamlit server is not the same as accessing a visitor's local webcam. The current camera workflow is designed to run on the computer connected to the camera.
- Model predictions are potential defects and must be visually verified.
- Real-world inspection performance requires testing with representative PCB samples.

## Future Improvements

- Evaluate and improve model performance on more diverse PCB samples.
- Optimize live inference speed and resource usage.
- Expand and improve the training dataset.
- Improve camera handling and the user interface.
- Explore additional deployment options.

---

## Notes for Contributors

- Keep trained model files in the `models/` directory and preserve the filenames expected by `app.py`.
- Do not commit virtual environments, temporary files, or large datasets unless they are intentionally required and permitted for redistribution.
- Check the dataset and model licences before sharing them publicly.
