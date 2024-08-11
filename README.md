
---

# Libras Hand Gesture Recognition Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.7-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-yellow.svg)

This project uses computer vision and machine learning techniques to recognize hand gestures in Libras (Brazilian Sign Language) and predict the corresponding letter in real-time. The application was developed using a Random Forest model, demonstrating how machine learning algorithms can be applied to solve practical pattern recognition problems.

## Features

- **Real-time detection:** Recognizes hand gestures captured by the webcam.
- **Letter prediction:** A Random Forest model trained to predict alphabet letters based on gestures.
- **Intuitive visualization:** Displays the recognized letter with emphasis and a bounding box around the detected hand.

## Prerequisites

Before running the project, you need to have the following installed:

- Python 3.8 or higher
- OpenCV
- MediaPipe
- Scikit-learn
- Joblib

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/LucasDAMASCENO26/libras-gesture-recognition.git
cd libras-gesture-recognition
pip install -r requirements.txt
```

## Code Explanation

The project consists of three main scripts: `processamento.py`, `Treinamento_Modelo.py`, and `deteccao.py`. Below is an overview of each:

### `processamento.py`

This script processes images of hand gestures from the `data_set` folder, which contains subfolders `training` and `test` organized by alphabet letters (excluding H, J, K, W, X, Y, Z). The script detects hand landmarks using MediaPipe, extracts the coordinates, and saves them along with corresponding labels as `.npy` files for later use.

### `Treinamento_Modelo.py`

This script loads the preprocessed data (`landmarks.npy` and `labels.npy`), reshapes it for the Random Forest model, encodes the labels as numerical values, and splits the data into training and test sets. After training the Random Forest model, it evaluates the model’s accuracy and saves the trained model as `random_forest_model.pkl`.

### `deteccao.py`

This script captures real-time video input from the webcam and uses the trained Random Forest model to predict the Libras alphabet letter based on detected hand gestures. It uses MediaPipe to detect hand landmarks and draw a bounding box around the detected hand. The predicted letter is displayed on the screen with improved visual aesthetics.

## Contact

Feel free to reach out if you have any questions or suggestions. You can contact me via LinkedIn: www.linkedin.com/in/lucas-santos-0245482b2  
Email: lucas.damasceno.santos@icen.ufpa.br

---

