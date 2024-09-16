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

## Requirements

To run the code, you need the following Python packages installed:

```bash
pip install pandas scikit-learn joblib
