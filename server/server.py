from flask import Flask, request, jsonify
from flask_cors import CORS
import utils_backend

app = Flask(__name__)
CORS(app)  

@app.route('/get_location_names')
def get_location_names():
    return jsonify({'locations': utils_backend.get_location_names()})

@app.route('/get_area_types')
def get_area_types():
    return jsonify({'area_types': utils_backend.get_area_types()})

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        total_sqft = float(request.form['total_sqft'])
        location = request.form['location']
        area_type = request.form['area_type']
        bedroom = int(request.form['bedroom'])
        bath = int(request.form['bath'])

        estimated_price = utils_backend.get_estimated_prices(
    location,
    total_sqft,
    bath,
    bedroom,
    area_type
)


        return jsonify({'estimated_price': estimated_price})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction")
    utils_backend.load_saved_artifacts()
    app.run(debug=True)
