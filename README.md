# Intel-Scene-Classification-via-CNN
A custom TensorFlow/Keras CNN pipeline for 6-class natural scene classification using the Intel Image Dataset, featuring dynamic data loading, augmentation, and complete evaluation metrics.


### Target Classes (6 Categories)
* 🏢 Buildings
* 🌲 Forest
* 🧊 Glacier
* 🏔️ Mountain
* 🌊 Sea
* 🏙️ Street

* ---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Dataset Management:** `kagglehub` API
* **Data Processing & Visualization:** NumPy, Matplotlib, Seaborn, scikit-learn

---

## 🏗️ Model Architecture
The custom CNN consists of:
1. **Input Layer:** `(150, 150, 3)` RGB images
2. **Data Augmentation:** Rescaling $[0, 1]$, random horizontal flip, random rotation ($10\%$), random zoom ($10\%$)
3. **Convolutional Blocks (x3):** 
   * `Conv2D` (32, 64, 128 filters with ReLU activation)
   * `MaxPooling2D` $(2 \times 2)$
4. **Classifier Head:**
   * `Flatten`
   * `Dense` (512 units, ReLU)
   * `Dropout` ($0.5$ for regularization)
   * `Dense` (6 units, Softmax output)

---

