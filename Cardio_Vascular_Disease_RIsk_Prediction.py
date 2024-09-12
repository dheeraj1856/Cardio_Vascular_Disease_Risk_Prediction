import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('CVD_cleaned.csv')

# --- Data Cleaning Section ---
# Mapping categorical variables to binary format for easier analysis
diabetes_mapping = {
    'No': 0,
    'No, pre-diabetes or borderline diabetes': 0,
    'Yes, but female told only during pregnancy': 1,
    'Yes': 1
}
data['Diabetes'] = data['Diabetes'].map(diabetes_mapping)

# One-hot encoding for categorical feature 'Sex'
data = pd.get_dummies(data, columns=['Sex'])

# Convert remaining categorical variables to binary format
binary_columns = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History', 'Exercise']
for column in binary_columns:
    data[column] = data[column].map({'Yes': 1, 'No': 0})

# Mapping ordinal variables
general_health_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Excellent': 4
}
data['General_Health'] = data['General_Health'].map(general_health_mapping)

# Create a new BMI category feature
bmi_mapping = {
    'Underweight': 0,
    'Normal weight': 1,
    'Overweight': 2,
    'Obesity': 3
}
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=['Underweight', 'Normal weight', 'Overweight', 'Obesity'])
data['BMI_Category'] = data['BMI_Category'].map(bmi_mapping).astype(int)

# Map age categories to numerical values
age_category_mapping = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
    '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9,
    '70-74': 10, '75-79': 11, '80+': 12
}
data['Age_Category'] = data['Age_Category'].map(age_category_mapping)

# Remove the 'Checkup' column since it was mapped to 'Checkup_Frequency'
data = data.drop(columns=['Checkup'])

# Remove duplicate rows
data = data.drop_duplicates()

# --- Data Visualization Section ---

# Univariate Analysis
# Plotting numerical features' distribution
numerical_features = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
                      'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

for feature in numerical_features:
    plt.figure(figsize=(10, 4))
    sns.histplot(data=data, x=feature, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Plotting categorical features' count
categorical_features = ['General_Health', 'Exercise', 'Sex_Male', 'Heart_Disease',
                        'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis']

for feature in categorical_features:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=data, x=feature)
    plt.title(f'Count of {feature}')
    plt.xticks(rotation=90)
    plt.show()

# Bivariate Analysis
# Analyzing relationships between selected variables and disease conditions
selected_variables = ['General_Health', 'Exercise', 'Sex_Male', 'Age_Category', 'Smoking_History']
disease_conditions = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Diabetes', 'Arthritis']

for disease in disease_conditions:
    for variable in selected_variables:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=data, x=variable, hue=disease)
        plt.title(f'Relationship between {variable} and {disease}')
        plt.xticks(rotation=90)
        plt.show()

# Multivariate Analysis
# Distribution of General Health by Age Category
plt.figure(figsize=(10, 7))
sns.countplot(data=data, x='General_Health', hue='Age_Category')
plt.title('Distribution of General Health by Age Category')
plt.xticks(rotation=90)
plt.show()

# BMI Category distribution by Exercise and disease conditions
for disease in disease_conditions:
    plt.figure(figsize=(10, 7))
    sns.countplot(data=data, x='BMI_Category', hue=disease)
    plt.title(f'Distribution of {disease} by BMI Category')
    plt.xticks(rotation=90)
    plt.show()

# 3D Plot - Age Category, General Health, and BMI
le = LabelEncoder()
data_3D = data[['Age_Category', 'General_Health', 'BMI']].copy()

# Encode categorical variables for 3D plotting
data_3D['Age_Category'] = le.fit_transform(data_3D['Age_Category'])
data_3D['General_Health'] = le.fit_transform(data_3D['General_Health'])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3D['Age_Category'], data_3D['General_Health'], data_3D['BMI'], c='blue', s=60)
ax.view_init(30, 185)
plt.xlabel("Age Category")
plt.ylabel("General Health")
ax.set_zlabel('BMI')
plt.title('3D plot of Age Category, General Health, and BMI')
plt.show()

# --- Data Modeling Section ---
# Define the features and target variable
X = data.drop('Heart_Disease', axis=1)
y = data['Heart_Disease']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline with resampling and scaling
resampling = SMOTE(sampling_strategy='minority')  # Oversample the minority class
tomek = TomekLinks(sampling_strategy='majority')  # Undersample the majority class
scaler = MinMaxScaler()  # Scale features

# Define the model with XGBoost
model = XGBClassifier(
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Adjust class imbalance
    eval_metric='logloss',  # Metric to evaluate model performance
    use_label_encoder=False  # Avoid warning
)

# Define a pipeline with scaling, oversampling, and the classifier
pipeline = Pipeline([('MinMaxScale', scaler), ('SMOTE', resampling), ('TomekLinks', tomek), ('Model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# --- Model Evaluation Section ---
# Make predictions on the test set
y_pred = pipeline.predict(X_test)
y_score = pipeline.predict_proba(X_test)[:, 1]

# Print classification report
print(classification_report(y_test, y_pred))

# Compute and display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

# Compute and plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Compute and plot the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()

# Plot learning curve
train_sizes, train_scores, valid_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, valid_scores.mean(axis=1), label='Cross-validation score')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.title('Learning curve')
plt.legend(loc="lower right")
plt.show()
