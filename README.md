# Breast Cancer Image Classification

This project aims to build a machine learning model to classify breast cancer images as either **benign** or **malignant**. The project involves data preparation, model building, training, and evaluation.

---

## Project Structure

Breast-Cancer-Image-Classification/ ├─ Detect_BreastCancer.ipynb ├─ Kaggle Link.txt ├─ output/ ├─ sampleTest_Pictures/ ├─ train_CustomModel_32_conv_20k.ipynb ├─ train_ResNet50_32_20k.ipynb └─ utils/ ├─ config.py ├─ conv_bc_model.py ├─ create_dataset.py └─ getPaths.py


### Key Components

1. **Data Preparation**  
   - **Kaggle Link**: The dataset is sourced from Kaggle, as indicated in `Kaggle Link.txt`.  
   - **Dataset Creation**: The script `create_dataset.py` organizes the dataset into training, validation, and testing splits.  
   - **Configuration**: Configuration settings (like paths, hyperparameters) are defined in `config.py`.

2. **Model Building**  
   - **Custom Model**: The custom convolutional neural network (CNN) model is defined in `conv_bc_model.py`. The `BC_Model` class builds a Sequential model with multiple convolutional, activation, batch normalization, pooling, dropout, and dense layers.  
   - **Training Scripts**:  
     - `train_CustomModel_32_conv_20k.ipynb`: Trains the custom CNN model.  
     - `train_ResNet50_32_20k.ipynb`: Trains a ResNet50 model.

3. **Model Training and Evaluation**  
   - **Training**: The training process involves loading the dataset, defining the model architecture, compiling the model, and fitting it to the training data. The training history and model weights are saved for later evaluation.  
   - **Evaluation**: The trained models are evaluated on the validation and testing datasets to measure their performance.

4. **Inference**  
   - **Detection Notebook**: `Detect_BreastCancer.ipynb` is used for making predictions on new images. It loads the trained model and uses it to classify sample images as benign or malignant.

---

## Detailed Explanation of Key Files

1. **`utils/config.py`**  
   Contains configuration settings such as paths, split ratios, batch size, and learning rate.

2. **`utils/create_dataset.py`**  
   Prepares the dataset by splitting it into training, validation, and testing sets and organizing the images into corresponding directories.

3. **`utils/conv_bc_model.py`**  
   Defines the custom CNN model architecture using TensorFlow and Keras.

4. **`train_CustomModel_32_conv_20k.ipynb`**  
   Jupyter notebook for training the custom CNN model.

5. **`train_ResNet50_32_20k.ipynb`**  
   Jupyter notebook for training the ResNet50 model.

6. **`Detect_BreastCancer.ipynb`**  
   Jupyter notebook for making predictions using the trained model.

---

## Steps to Run the Project

### 1. Download the Dataset
- Download the dataset from Kaggle as indicated in `Kaggle Link.txt`.

### 2. Prepare the Dataset
- Run the script `create_dataset.py` to organize the dataset into training, validation, and testing splits.

### 3. Train the Model
- Use **`train_CustomModel_32_conv_20k.ipynb`** to train the custom CNN model.
- Alternatively, use **`train_ResNet50_32_20k.ipynb`** to train the ResNet50 model.

### 4. Evaluate the Model
- Evaluate the trained models on the validation and testing datasets to measure their performance.

### 5. Make Predictions
- Use **`Detect_BreastCancer.ipynb`** to make predictions on new images.

---

## Configuration

- **Paths**: Set the paths for the dataset and output directories in `config.py`.
- **Hyperparameters**: Adjust the batch size, learning rate, and other hyperparameters in `config.py`.

---

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## Installation

Install the required packages using pip:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
