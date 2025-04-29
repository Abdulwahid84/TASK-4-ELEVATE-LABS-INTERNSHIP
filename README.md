# TASK-4-ELEVATE-LABS-INTERNSHIP
# Binary Classification with Logistic Regression

## 📌 Objective
This project demonstrates how to build a **binary classifier** using **Logistic Regression**. It includes data preprocessing, model training, evaluation using various metrics, and an exploration of the decision threshold and the sigmoid function.

---

## 🛠️ Tools and Libraries Used
- Python 3.x
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning modeling
- [Matplotlib](https://matplotlib.org/) for data visualization

---

## 📊 Dataset
We use a binary classification dataset (e.g., Breast Cancer dataset from `sklearn.datasets`), which has features and a binary target (0 or 1). You can easily swap in another dataset as needed.

---

## 🔍 Workflow

### 1. Data Preparation
- Load dataset and select relevant features
- Split the data into training and testing sets using `train_test_split`
- Standardize the features using `StandardScaler`

### 2. Model Training
- Fit a `LogisticRegression` model on the training set

### 3. Evaluation
- Evaluate the model using:
  - **Confusion Matrix**
  - **Precision**
  - **Recall**
  - **ROC Curve and AUC Score**

### 4. Threshold Tuning
- Adjust the default decision threshold (0.5) to observe its impact on precision and recall
- Visualize the trade-off using a precision-recall curve

### 5. Understanding the Sigmoid Function
- Plot the sigmoid function
- Explain how logistic regression maps inputs to probabilities using the sigmoid

---

## 📈 Visuals
- Confusion Matrix Heatmap
- ROC Curve
- Precision-Recall Curve
- Sigmoid Function Plot

---

## 📂 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/logistic-regression-classifier.git
   cd logistic-regression-classifier
