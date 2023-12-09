from flask import Flask, render_template, request, jsonify
from chatbot_text_image_audio import Chatbot_Response
import speech_recognition as sr
import requests


app = Flask(__name__)

# Initialize the chatbot
chatbot = Chatbot_Response()  

@app.route('/')
def home():
    return render_template('index.html')

import base64
import os
import tempfile
from flask import Flask, jsonify, request

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    image_data = request.form.get('image_data')

    response_message = ""
    image_html = ""
    temp_image_path = None
    is_image_relevant = True  # Assume image is relevant by default
    # handle zoom guardrail 
    if "zoom" in message.lower():
        return jsonify({
        "user_message": message,
        "bot_response": "At this time, we don't have this information. Please contact the admin.",
        "image_html": image_html
    })
    try:
        if image_data:
            # Assume image_data is already in base64 format for HTML display
            image_html = f'<img src="{image_data}" />'
            header, image_data = image_data.split(',', 1) if ',' in image_data else ('', image_data)
            image_bytes = base64.b64decode(image_data)
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
                temp_image_path = temp_image.name
                temp_image.write(image_bytes)

            # Call your function with the image path
            response_message,is_image_relevant = chatbot.answer_question_with_image(message, temp_image_path)
            if not is_image_relevant:
                # If image is not relevant, set image_html to None
                image_html = None
            else:
                # If image is relevant, include it in the response
                image_html = f'<img src="data:image/png;base64,{image_data}" />'
        else:
            response_message = chatbot.answer_question(message)     
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)  # Clean up the temp file
        

    # Send back the user message, chatbot response, and image HTML if present
    return jsonify({
        "user_message": message,
        "bot_response": response_message,
        "image_html": image_html
    })

@app.route('/send_audio', methods=['POST'])
def audio():
    response_message = ""
    try:
        if 'audio' not in request.files:
            return {'error': 'No audio file provided'}, 400

        audio_file = request.files['audio']

        # Open the file only once
        with open('recording.wav', 'wb') as f:
            f.write(audio_file.read())

        audio_path= 'recording.wav'

        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-base"
        headers = {"Authorization": f"Bearer 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"}

        with open(audio_path, "rb") as f:
            data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)
        
        message= response.json()['text'].strip()
        print(message) 
        response_message = chatbot.answer_question(message) 

    except Exception as e:
        return {'error': str(e)}, 500
    
    # Send back the user message, chatbot response, and image HTML if present
    return jsonify({
        "user_message": message,
        "bot_response": response_message
    })



if __name__ == "__main__":
    app.run(debug=True)
