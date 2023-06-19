# Import libraries
import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow import keras
import tensorflow as tf
import pyttsx3
from googletrans import Translator


# Load the pre-trained model from the 'modelWeights.h5' file
model = keras.models.load_model('modelWeights.h5')

# Define an array of actions or labels that the model can predict
actions = np.array([
    'a', 'b', 'c',
    '0', '1', '2', '3',
    'hello', 'goodbye', 'please', 'thank you',
    'yes', 'no', 'help',
    'red', 'blue', 'green',
    'mother', 'father',
    'eat', 'drink',
    'welcome', 'to', 'our', 'graduation', 'project', 'discussion',
    'idle'
])

# Initialize the holistic model from MediaPipe
mp_holistic = mp.solutions.holistic

# Initialize the drawing utilities from MediaPipe
mp_drawing = mp.solutions.drawing_utils


########################################
#########  Auxilary functions  #########
########################################


def play_word(word):
    """
    Function to play the given word using text-to-speech.
    """
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Say the word
    engine.say(word)

    # Wait for the speech to finish
    engine.runAndWait()


def translate_word(text, target_lang):
    """
    Function to translate the given text to the target language.
    """
    # Create an instance of the Translator class
    translator = Translator()

    # Translate the text to the target language
    translation = translator.translate(text, dest=target_lang)

    # Get the translated text
    translated_text = translation.text

    # Return the translated text
    return translated_text


def mediapipe_detection(image, model):
    """
    Function to perform object detection using MediaPipe.
    """
    # Convert the image to RGB color format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Disable writeable flag to improve performance
    image.flags.writeable = False

    # Process the image using the specified model
    results = model.process(image)

    # Enable writeable flag to make the image writable again
    image.flags.writeable = True

    # Convert the image back to BGR color format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Return the processed image and the results
    return image, results


def extract_keypoints(results):
    # Extract pose keypoints if available, otherwise create a zero-filled array
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)

    # Extract right hand keypoints if available, otherwise create a zero-filled array
    righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)

    # Extract left hand keypoints if available, otherwise create a zero-filled array
    lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)

    # Concatenate pose, right hand, and left hand keypoints into a single array
    return np.concatenate([pose, righthand, lefthand])


def draw_landmarks(image, results):
    # Draw pose landmarks on the image with specific color and thickness
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1))
    # Draw left hand landmarks on the image with specific color and thickness
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1))
    # Draw right hand landmarks on the image with specific color and thickness
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1))


def add_line_breaks(paragraph, line_length):
    """
    Function to add line breaks to a paragraph based on a given line length.
    """
    lines = []
    current_line = ""
    words = paragraph.split()

    for word in words:
        if len(current_line) + len(word) <= line_length:
            current_line += word + " "
        else:
            lines.append(current_line.rstrip())  # Remove trailing whitespace
            current_line = word + " "

    lines.append(current_line.rstrip())  # Remove trailing whitespace
    return "\n".join(lines)


#############################################
#########  Streamlit page building  #########
#############################################

# Open video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# List of languages for translation
langs = ['en', 'ar', 'fr', 'es', 'de']

# Set the title of the Streamlit app
st.title("Sign Language Recogniser")

# Placeholder to display video frames
frame_placeholder = st.empty()

# Initialize an empty paragraph string
paragraph = ""

# Placeholder to display the recognized text
text_placeholder = st.empty()

# Dropdown to select the language for translation
language = st.selectbox('Select language', langs)

# Display options checkboxes
st.write('Select the options you want')
landMarkBool = st.checkbox('Draw landmarks')
ttsBool = st.checkbox('Text-to-Speech')
debug = st.checkbox('Debug Mode')

#Log Out Button
if st.button("LogOut"):
    f = open("vars.txt", "a")
    f.write(str(0))
    f.close()
    
    
#############################################
#########   Model and Prediction    #########
#############################################


# Initialize lists for storing sequence and sentence
sequence = []
sentence = []

# Set the threshold for prediction confidence
threshold = 0.5

# Initialize the Holistic model from the Mediapipe library with specified confidence thresholds
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Start a loop to process each frame from the video capture
    while cap.isOpened():
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Make detections using the Holistic model
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks on the frame if landMarkBool is True
        if landMarkBool:
            draw_landmarks(image, results)

        # Prediction logic
        if len(sequence) == 0:
            # Display a message to indicate that the next word is being taken
            cv2.putText(image, 'TAKING NEXT WORD', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

            # Convert the image to RGB and display it in the Streamlit app
            frame1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame1, channels="RGB")
            cv2.waitKey(2000)

        # Extract keypoints from the detection results
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-45:]

        if len(sequence) == 45:
            # Perform prediction using the model
            res = model.predict(np.expand_dims(
                sequence, axis=0), verbose=None)[0]
            pred = actions[np.argmax(res)]

            # Append the predicted action to the paragraph
            paragraph += translate_word(pred, language)
            paragraph = add_line_breaks(paragraph, 70)
            paragraph += " "

            # Update the displayed text in the Streamlit app
            text_placeholder.text(paragraph)

            # Play the word using text-to-speech if ttsBool is True
            if ttsBool:
                play_word(pred)

            sequence = []
            sentence.append(pred)

            # Visualize the last 5 predicted words
            if len(sentence) > 5:
                sentence = sentence[-5:]

        # Draw a rectangle and display the current sentence on the frame
        if debug:
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Check if video capture has ended
        if not ret:
            st.write("Video Capture has ended")
            break

        # Convert the image to RGB and display it in the Streamlit app
        frame1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame1, channels="RGB")

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()