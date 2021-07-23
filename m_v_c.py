from flask import Flask, jsonify, request

from classifier import get_pred

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])

def pred_data():
    image =  request.files.get("alphabet")
    pred = get_pred(image)
    return jsonify({
        'Prediction': pred
    }), 200

if (__name__ == "__main__"):
    app.run(debug = True)