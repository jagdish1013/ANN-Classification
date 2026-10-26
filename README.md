# ANN Classification – Date Fruit Dataset

This project implements an **Artificial Neural Network (ANN)** for multiclass classification of different varieties of date fruits using **PyTorch**.

## 📌 Project Overview

The model uses numerical features extracted from date fruit samples to classify them into one of **7 different date fruit classes**.

The dataset contains **898 samples and 35 columns**, including **34 input features** and one target column named `Class`.

### Date Fruit Classes

The model classifies the following seven varieties:

* BERHI
* DEGLET
* DOKOL
* IRAQI
* ROTANA
* SAFAVI
* SOGAY

The target labels are converted into numerical values using `LabelEncoder`.

---

## 🧠 Model Architecture

The ANN is implemented using **PyTorch**.

The network consists of:

```text
Input Layer
     ↓
Linear Layer → 64 neurons
     ↓
ReLU
     ↓
Linear Layer → 64 neurons
     ↓
ReLU
     ↓
Output Layer → 7 classes
```

The model architecture is:

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(x.shape[1], 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 7)
        )

    def forward(self, x):
        return self.model(x)
```

The output layer contains 7 neurons because the dataset contains 7 target classes.

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Separate features and target variable.
3. Encode categorical class labels using `LabelEncoder`.
4. Split the dataset into training and testing sets.
5. Standardize the features using `StandardScaler`.
6. Convert the data into PyTorch tensors.
7. Create PyTorch `TensorDataset` and `DataLoader` objects.

The dataset is divided using:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

The features are standardized using `StandardScaler`.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Jupyter Notebook

### Main Libraries

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

---

## 🏋️ Training

The ANN is trained for **100 epochs** using:

* **Optimizer:** Adam
* **Loss Function:** Cross Entropy Loss
* **Batch Size:** 32
* **Epochs:** 100

```python
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```

The training loop performs forward propagation, loss calculation, backpropagation, and parameter updates.

The recorded training loss decreased from approximately **1.12 at the beginning of training to 0.024 by epoch 99**.

---

## 📊 Model Performance

The model was evaluated on **180 test samples**.

### Results

| Metric              |     Result |
| ------------------- | ---------: |
| Test Samples        |        180 |
| Correct Predictions |        170 |
| Accuracy            | **94.44%** |

The notebook's evaluation produced an accuracy of **94.4444%**.

---

## 📁 Project Structure

```text
ANN-Classification/
│
├── ANN_Clasification.ipynb
├── DateFruit_Dataset.csv
└── README.md
```

> Note: The dataset file is required because the notebook loads it using `pd.read_csv("DateFruit_Dataset.csv")`.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/jagdish1013/ANN-Classification.git
```

### 2. Open the project

```bash
cd ANN-Classification
```

### 3. Install the required libraries

```bash
pip install pandas numpy scikit-learn torch jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
ANN_Clasification.ipynb
```

Make sure `DateFruit_Dataset.csv` is in the same directory as the notebook.

---

## 📈 Results

The ANN achieved:

**94.44% test accuracy**

with:

* 898 total samples
* 34 numerical input features
* 7 classification categories
* 80/20 train-test split
* 100 training epochs
* Batch size of 32

---

## 🔮 Future Improvements

Possible improvements to this project include:

* Hyperparameter tuning
* Dropout for regularization
* Early stopping
* Learning-rate scheduling
* Confusion matrix visualization
* Precision, recall, and F1-score
* Training/validation loss graphs
* Model checkpointing
* Saving the trained PyTorch model



GitHub:
https://github.com/jagdish1013

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.


