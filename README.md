# Fire and Smoke Detection with YOLOv8

A deep learning-based fire and smoke detection system using YOLOv8, featuring both a command-line testing script and a Streamlit web interface.

## Overview

This project fine-tunes a YOLOv8 model on a custom fire and smoke dataset to detect fires and smoke in images and real-time video. The system includes:

- **Test script** (`test.py`): Command-line detection on single images
- **Web interface** (`app.py`): Interactive Streamlit app for image upload and camera detection
- **Pre-trained model**: Fine-tuned YOLOv8s weights available in `runs/detect/fire_smoke_optimized1/weights/`

## Results

The model achieves good detection performance on fire and smoke objects:
- **Test image (`test1.png`)**: Detects 3 fires and 1 smoke with confidence threshold 0.25
- **Model metrics**: Precision, Recall, mAP metrics available in training results

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ultralytics-main
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Command-line Testing

Run detection on a test image:
```bash
python test.py
```

The script will load the trained model (`best.pt`) and perform detection on `test1.png`. Results are displayed and saved.

#### 2. Web Interface

Launch the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

**Features**:
- Upload images for detection
- Real-time camera detection (requires webcam)
- Adjustable confidence threshold (default: 0.25)
- Device selection (CPU/GPU)
- Display of training metrics and charts

## Project Structure

```
.
├── app.py                  # Streamlit web interface
├── test.py                 # Command-line test script
├── requirements.txt        # Python dependencies
├── runs/detect/fire_smoke_optimized1/  # Trained model and results
│   ├── weights/
│   │   ├── best.pt        # Best model weights
│   │   └── last.pt        # Last checkpoint weights
│   ├── results.csv        # Training metrics
│   └── *.png              # Training charts
├── test1.png              # Sample test image 1
├── test2.png              # Sample test image 2
└── fire-smoke/            # Training dataset (optional)
```

## Model Training

The model was trained on a custom fire and smoke dataset with the following configuration:
- **Model**: YOLOv8s
- **Epochs**: 150
- **Classes**: Fire (0), Smoke (1)
- **Image size**: 640x640
- **Dataset**: Custom collected and annotated images

Training results are saved in the `runs/detect/fire_smoke_optimized1` directory.

## Configuration

### Confidence Threshold

The detection sensitivity can be adjusted via the confidence threshold:
- **Lower values (0.1-0.3)**: More detections, higher recall, potential false positives
- **Higher values (0.5-0.7)**: Fewer detections, higher precision, potential missed objects

Default is set to 0.25 to balance precision and recall.

### Device Selection

- **CPU**: Default when GPU is unavailable
- **GPU**: Automatically selected if compatible CUDA device is available

## Troubleshooting

### CUDA Compatibility Issues

If you encounter CUDA errors (e.g., "no kernel image is available"), force CPU usage:

1. In `test.py`, modify the model call:
   ```python
   results = model('test1.png', device='cpu')
   ```

2. In the web interface, select "CPU" from the device dropdown.

### Installation Issues

Ensure you have the correct PyTorch version for your system. If needed, install PyTorch separately:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

## License

This project is based on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base model and framework
- Contributors of the fire and smoke dataset

## Contact

For questions or issues, please open an issue in the repository.