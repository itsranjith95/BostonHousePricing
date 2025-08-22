import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']   # expecting JSON {"data": {...}}
    print("Input Data:", data)

    # Convert input dictionary to numpy array
    new_data = np.array(list(data.values())).reshape(1, -1)

    # Use the pre-fitted scaler
    scaled_data = scaler.transform(new_data)

    # Predict using the regression model
    output = regmodel.predict(scaled_data)
    print("Prediction:", output[0])

    return jsonify({"prediction": float(output[0])})

if __name__ == "__main__":
    app.run(debug=True)
