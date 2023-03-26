from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('braille_model2.h5')
characters = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
img_size = (32, 32)

app = Flask(__name__)


@app.route('/braille', methods=['POST'])
def convert_to_braille():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load the image
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(img, img_size)
        image = np.reshape(image, (img_size[0], img_size[1], 1))
    except Exception as e:
        return jsonify({'error': 'Error opening image file: ' + str(e)}), 400


    # Make a prediction using the loaded model
    prediction = model.predict(image)
    output_label = ''.join([characters[i] for i in prediction])

    # Return the converted text as a response
    response = {'converted_text': output_label}
    return jsonify(response), 200

@app.route('/')
def welcome():
    return "Welcome to this braille to text API."

if __name__ == '__main__':
    app.run(debug=True)
