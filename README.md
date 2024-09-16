# Music Genre Prediction

This project uses a Decision Tree Classifier from `scikit-learn` to predict music genres based on user data. The dataset is taken from Kaggle, and the model is trained and tested using this data. The project involves splitting the data, training the model, testing it, and checking its accuracy.

## Dataset

The dataset used in this project is from Kaggle and can be found here:

[Kaggle Music Dataset](https://www.kaggle.com/c/your-dataset-link)

### Features:
- Various attributes (age, gender, etc.)
- Target variable: `genre` (the music genre)

## Project Steps

### 1. Load the Data
We start by loading the dataset using `pandas` and then split the data into features (`X`) and labels (`y`). The `X` represents all the columns except `genre`, and `y` represents the `genre` column.

### 2. Split the Data
We divide the data into two parts: 
- **Training Set** (80% of the data)  
- **Test Set** (20% of the data)

This split helps us train the model on one part and evaluate it on another.

### 3. Train the Model
We use the `DecisionTreeClassifier` from `scikit-learn` to build the model. The model is trained on the training data (`X_train`, `y_train`).

### 4. Save the Model
The trained model is saved using the `joblib` library for future predictions without retraining.

### 5. Test and Evaluate the Model
Once the model is trained, we test it using the test data (`X_test`, `y_test`) to see how well it performs. The accuracy of the model is calculated using `accuracy_score`.

# Music Genre Prediction with Decision Tree Classifier

This project demonstrates how to use a Decision Tree Classifier to predict music genres based on user data. It involves splitting the data, training the model, saving it, and testing its accuracy.

## Project Steps

1. **Load the Dataset:** We load the dataset using `pandas` from a CSV file.
2. **Split the Data:** The dataset is divided into features (`X`) and labels (`y`). We further split the data into training and testing sets.
3. **Train the Model:** We use a `DecisionTreeClassifier` to train the model on the training data.
4. **Save the Model:** The trained model is saved using `joblib` for future use.
5. **Test the Model:** We make predictions on the test data and calculate the accuracy score.

## Code Implementation

Here’s the Python code that implements the above steps:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
music_data = pd.read_csv('music.csv')

# Split the data into features and labels
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the model and fit it to the training data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'music-recommender.joblib')

# Make predictions on the testing set
predictions = model.predict(X_test)

# Calculate the accuracy score
score = accuracy_score(y_test, predictions)
print(f"Accuracy: {score}")


