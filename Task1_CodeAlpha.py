# 1. Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load dataset from downloaded CSV
df = pd.read_csv('Iris.csv')

# 3. View first few rows
print("First 5 rows of the dataset:")
print(df.head())

# 4. Drop ID column (not useful for prediction)
df.drop('Id', axis=1, inplace=True)

# 5. Display basic info and class distribution
print("\nDataset Info:")
print(df.info())

print("\nClass distribution:")
print(df['Species'].value_counts())

# 6. Visualize the data
sns.pairplot(df, hue='Species')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# 7. Split data into features and target
X = df.drop('Species', axis=1)
y = df['Species']

# 8. Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train the model using Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 10. Predict and evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=df['Species'].unique(),
            yticklabels=df['Species'].unique())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. (Optional) Save the model
import joblib
joblib.dump(model, 'iris_classifier_model.pkl')
print("\nModel saved as 'iris_classifier_model.pkl'")
