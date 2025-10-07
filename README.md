🏠 House Price Prediction using Machine Learning

This project predicts house prices based on key property features using Linear Regression.
It was trained on data4.csv and includes feature selection, model training, evaluation, and real-time predictions with input validation.

✨ Features

📊 Data Preprocessing & Feature Selection

Uses SelectKBest (f_regression) to automatically pick the top 5 most relevant features.

🧮 Model Training

Implements Linear Regression on the selected features.

Displays model coefficients and intercept for transparency.

🧪 Evaluation

Reports R² scores for both training and test sets.

🖥️ User Input & Prediction

CLI-based interface to input house details.

Includes input validation:

No negative values

Logical checks (e.g., max rooms ≤ 50)

Outputs predicted house price in a clean format.

🌐 Basic Web Version

A simple web interface has also been created.

🚀 A more polished UI will be built soon.

🛠️ Tech Stack

Python 3

Pandas – data handling

Scikit-learn – ML model & feature selection
