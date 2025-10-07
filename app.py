from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --- Load dataset and train model (exactly your original code) ---
df2 = pd.read_csv("data4.csv")

X = df2.drop(columns=['price'])
y = df2['price']

selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
X = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

# --- Home page ---
@app.route('/')
def home():
    return render_template('index.html', features=selected_features.tolist())

# --- Predict endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # get input from frontend
        user_input = {}

        for feature in selected_features:
            value = float(data.get(feature, None))

            # same validation as your original code
            if value < 0:
                return jsonify({"error": f"❌ Invalid input: {feature} cannot be negative."})
            if "room" in feature.lower() and value > 50:
                return jsonify({"error": f"❌ Invalid input: {feature} seems too large."})

            user_input[feature] = [value]

        user_df = pd.DataFrame(user_input)
        predicted_price = model.predict(user_df)[0]

        return jsonify({
            "predicted_price": f"${predicted_price:,.2f}",
            "train_r2": round(train_r2, 3),
            "test_r2": round(test_r2, 3),
            "selected_features": selected_features.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
