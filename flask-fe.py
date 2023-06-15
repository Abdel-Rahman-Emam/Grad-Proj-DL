from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import import_ipynb
from nbimporter import find_notebook
find_notebook('Untitled')

app = Flask(__name__)
sequence = []
sentence = []
predictions = []
threshold = 0.5

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = tf.keras.models.load_model('action.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the latest video frame
    image_data = request.files['image'].read()
    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Make detections
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(image, holistic)

    # Draw landmarks
    draw_landmarks(image, results)

    # Prediction logic
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))

        if np.array_equal(np.unique(predictions[-10:]), np.argmax(res)):
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        # Viz probabilities
        image = prob_viz(res, actions, image, colors)

    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Encode the image as a JPEG file for transmission
    ret, jpeg = cv2.imencode('.jpg', image)
    image_data = jpeg.tobytes()

    # Convert the prediction to a JSON response
    response = {'prediction': ' '.join(sentence)}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)