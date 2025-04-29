#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report, precision_score, recall_score
     


# In[2]:


df = pd.read_csv("Breast Cancer Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.size


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.dtypes


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df=df.drop(['Unnamed: 32'],axis=1)


# In[13]:


df.shape


# In[14]:


for i in df.columns:
    print(df[i].unique())
    print('-'*100)
    print(df[i].value_counts())
    print('*'*100)


# In[17]:


df.duplicated().sum()


# In[19]:


df.isna().sum()


# In[20]:


##Seperate X and Y for training.


# In[21]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Encode diagnosis to numerical values


# In[22]:


# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()


# In[23]:


# Create transformers
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[24]:


# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])


# In[25]:


# Create and train the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[26]:


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

# Plot the confusion matrix
disp.plot()
plt.show()

# Extract TP, TN, FP, FN
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")
print(f"True Positive (TP): {TP}")


# In[27]:


x = df.drop("diagnosis",axis=1)
y = df["diagnosis"]


# In[28]:


x.head()


# In[29]:


y.head()


# In[30]:


# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)


print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC Score: {roc_auc}")

# Calculate ROC curve
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[31]:


# Loops through the thresholds and find the best threshold
best_threshold = 0.5
best_f1_score = 0

for threshold in np.arange(0.1, 1, 0.05):
  y_pred_threshold = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)
  report = classification_report(y_test, y_pred_threshold, output_dict=True)
  f1 = report['macro avg']['f1-score']  # Using macro-average F1-score for demonstration
  if f1 > best_f1_score:
      best_f1_score = f1
      best_threshold = threshold

print(f"Best Threshold: {best_threshold}")

y_pred_best = (model.predict_proba(X_test)[:,1]>best_threshold).astype(int)
print(classification_report(y_test, y_pred_best))

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.title('Precision-Recall Curve')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
plt.legend()
plt.grid(True)
plt.show()

