# 🖊️ Handwritten Digit Prediction

This project predicts handwritten digits using the **scikit-learn digits dataset**. The dataset consists of **8x8 grayscale images** of handwritten digits (0-9). A machine learning model is trained to classify these digits based on their pixel values.

## 📌 Features
- Loads and visualizes the handwritten digits dataset.
- Preprocesses the dataset using **StandardScaler**.
- Splits data into training and testing sets.
- Trains a **Logistic Regression** model for digit classification.
- Evaluates model performance using **confusion matrix** and **accuracy score**.

## 🛠️ Technologies Used
- **Python** 🐍
- **scikit-learn** (for ML model)
- **Matplotlib & Seaborn** (for data visualization)
- **Pandas & NumPy** (for data handling)

## 📂 Dataset
This project uses the built-in **digits dataset** from `sklearn.datasets`. The dataset contains:
- **1,797 images** of size **8×8 pixels**.
- Grayscale values ranging from **0 to 16**.
- Labels corresponding to **digits 0-9**.

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/handwritten-digit-prediction.git
cd handwritten-digit-prediction
```
## 2️⃣ Install Dependencies
```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```
## 3️⃣ Run the Notebook
Open Jupyter Notebook and run:

```sh
jupyter notebook "Handwritten Digit Prediction.ipynb"
```
## 📊 Model Training & Evaluation
- The dataset is split into 80% training and 20% testing.
- A Logistic Regression model is trained on the training set.
- Evaluation Metrics:
    - Confusion Matrix for class-wise performance.
    - Classification Report for precision, recall, and F1-score.
    - Overall Accuracy Score.
## 📸 Visualization Example
The dataset contains handwritten digit samples like these:



### 🔮 Future Improvements
Implement a Deep Learning Model (CNN) for improved accuracy.
Experiment with other models like SVM, KNN, Random Forest.
Deploy the trained model using a Flask or FastAPI web app.
