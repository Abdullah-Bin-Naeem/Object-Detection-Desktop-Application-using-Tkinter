# Object Detection Desktop Application using Tkinter

This repository contains a desktop application built using **Tkinter** for performing object detection tasks. The app supports multiple models, including **YOLOv11** and **Faster R-CNN**, trained on different datasets for various use cases. Below is a detailed guide to help you understand the structure, functionality, and usage of this project.

---

### Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Models and Datasets](#models-and-datasets)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training Custom Models](#training-custom-models)
7. [Requirements](#requirements)
8. [Folder Structure](#folder-structure)
9. [Contributing](#contributing)
10. [License](#license)

---

### Overview
The application provides a user-friendly interface for performing object detection on images. Users can:
- Select a pre-trained model from a dropdown menu.
- Upload an image for inference.
- View the detected objects with bounding boxes and labels.

The app is designed to be modular, allowing easy integration of additional models or datasets in the future.

---

### Features
- **Model Selection**: Choose between YOLOv11 and Faster R-CNN models.
- **Image Upload**: Upload an image file for inference.
- **Object Detection**: Detect objects in the uploaded image based on the selected model.
- **Customizable Models**: Easily train new models by modifying the provided training notebooks.
- **Support for Multiple Use Cases**:
  - YOLOv11: Self-driving dataset (e.g., cars, pedestrians) and UNO card detection.
  - Faster R-CNN: Synthetic fruit detection and UNO card detection.

---

### Models and Datasets
#### Pre-trained Models
1. **YOLOv11**:
   - Trained on:
     - **Self-driving dataset** (cars, pedestrians, etc.)
     - **UNO cards and numbers**
   - Training notebook: `yolo_training.ipynb`
   - Configuration: Modify `data.yaml` for custom datasets.

2. **Faster R-CNN**:
   - Trained on:
     - **Synthetic fruits dataset**
     - **UNO cards and numbers**
   - Training notebook: `faster_rcnn_training.ipynb`

#### Datasets
- All datasets are sourced from **Roboflow**.
- You can replace the datasets in the training notebooks to train the models on your desired data.

---

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/object-detection-app.git
   cd object-detection-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained models and place them in the `models/` directory.

4. Ensure you have the following libraries installed:
   - OpenCV (`cv2`)
   - PyTorch
   - Tkinter

---

### Usage
1. Run the application:
   ```bash
   python app.py
   ```

2. In the GUI:
   - Select a model from the dropdown menu.
   - Click the "Upload Image" button to select an image file.
   - Click "Run Inference" to detect objects in the image.

3. The output will display the image with bounding boxes and labels for detected objects.

---

### Training Custom Models
To train your own models:
1. Open the respective training notebook:
   - For YOLOv11: `yolo_training.ipynb`
   - For Faster R-CNN: `faster_rcnn_training.ipynb`

2. Replace the dataset URL in the notebook with your desired dataset from Roboflow.

3. For YOLOv11:
   - Update the `data.yaml` file with the paths to your dataset.

4. Train the model and save the weights.

5. Place the trained weights in the `models/` directory and update the app code if necessary.

---

### Requirements
- Python 3.8+
- OpenCV (`cv2`)
- PyTorch
- Tkinter
- YOLOv11 (custom implementation)
- Faster R-CNN (via PyTorch)

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

### Folder Structure
```
object-detection-app/
├── app.py                     # Main application script
├── models/                    # Directory for pre-trained models
│   ├── yolo_weights.pth       # YOLOv11 weights
│   ├── faster_rcnn_weights.pth # Faster R-CNN weights
├── notebooks/                 # Training notebooks
│   ├── yolo_training.ipynb    # YOLOv11 training notebook
│   ├── faster_rcnn_training.ipynb # Faster R-CNN training notebook
├── data/                      # Dataset configuration files
│   ├── data.yaml              # YOLOv11 dataset configuration
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

### Contributing
We welcome contributions! If you'd like to add new models, improve the UI, or fix bugs:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Contact
For questions or feedback, please contact:
- Email: abdullahbinnaeempro@gmail.com
- LinkedIn: [Abdullah Bin Naeem]((https://www.linkedin.com/in/abdullah-bin-naeem)

--- 

Happy coding! 🚀
