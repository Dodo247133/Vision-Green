from flask import Flask, request, jsonify
from model.inference import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Save the image to a temporary location
        image_path = f'temp_{file.filename}'
        file.save(image_path)

        # Perform prediction
        predictions = predict(image_path)

        # Return the predictions as JSON
        return jsonify(predictions)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
