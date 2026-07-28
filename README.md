# Speech Keyword Spotting Model

This repository contains the implementation of a Deep Learning model designed to recognize short spoken commands using the Mini Speech Commands dataset. The project demonstrates a complete pipeline from audio signal processing to training a Convolutional Neural Network (CNN) for audio classification.

## 📌 Project Overview

Keyword spotting is a fundamental technology behind voice-activated systems (e.g., smart home assistants, robotics). The goal of this project is to classify one-second audio clips into one of eight distinct commands:
`['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']`

The project consists of two main components:

1.  **Training Pipeline (`train_model.py`):** Handles data downloading, preprocessing (converting waveforms to spectrograms), model architecture definition, training, and evaluation.
2.  **Inference Script (`use_model.py`):** Loads the trained model to predict the command spoken in a new, user-provided `.wav` file.

## ⚙️ Methodology & Architecture

### Data Preprocessing

- **Spectrogram Conversion:** Raw audio waveforms are converted into spectrograms using the Short-Time Fourier Transform (STFT). This 2D representation (time vs. frequency) allows the model to process audio data similarly to images.
- **Parallel Processing:** `tf.data.Dataset` with `AUTOTUNE` is utilized for efficient parallel loading and preprocessing of audio files.

### Model Architecture (CNN)

A Sequential Convolutional Neural Network built with Keras is used, which includes:

- **Resizing & Normalization:** Standardizing the spectrogram inputs (resized to 32x32).
- **Convolutional Layers (Conv2D):** Two layers (32 and 64 filters) to extract local frequency patterns.
- **MaxPooling & Dropout:** To reduce dimensionality and mitigate overfitting (Dropout rates of 0.25 and 0.5).
- **Dense Layers:** A fully connected layer with 128 units leading to the final classification layer.

### Optimization

- **Optimizer:** Adam.
- **Loss Function:** Sparse Categorical Crossentropy.
- **Callbacks:** Early Stopping (patience=2) is implemented to halt training when validation performance plateaus.

## 📊 Results & Evaluation

### Confusion Matrix Analysis

![Confusion Matrix](images/confusion_matrix.png)

- **High Accuracy:** The model performs exceptionally well on commands like `"stop"` (108 correct), `"yes"` (105 correct), and `"right"` (102 correct).
- **Challenges:** There is slight confusion between certain commands, such as `"down"` and `"go"`, or `"no"`, likely due to phonetic similarities.

### Sample Predictions

![Sample Predictions](images/prediction_sample.png)

The `use_model.py` script successfully outputs the probability distribution of commands for unseen audio files, demonstrating practical inference capabilities.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SudeZarei/Speech-Keyword-Spotting-Model.git](https://github.com/SudeZarei/Speech-Keyword-Spotting-Model.git)
    cd Speech-Keyword-Spotting-Model
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the model (Optional - the model downloads the dataset automatically):**
    ```bash
    python train_model.py
    ```
4.  **Run Inference on a new audio file:**
    - Create a folder named `voices` in the root directory, and inside it, create a subfolder with your preferred name (e.g., `voices/test/`).
    - Place your `.wav` audio clips inside that folder.
    - Update the `sample_file` path in `use_model.py` to point to your audio file.
    - Run the script:
      ```bash
      python use_model.py
      ```

---

_This project was developed by Sude Zarei, Fatemeh Farhadi, and Alireza Salehi Mehara for the Speech and Language Processing Fundamentals course at Hamedan University of Technology._
