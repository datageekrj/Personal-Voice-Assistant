from flask import Flask, render_template, request, jsonify, send_file
import json
import google.generativeai as genai
from io import BytesIO
from elevenlabs import play
from elevenlabs.client import ElevenLabs

app = Flask(__name__)
with open("api_key.json", "r") as file:
    data = json.load(file)

genai.configure(api_key=data["key"])
client = ElevenLabs(
  api_key=data["eleven_labs"],
)
model = genai.GenerativeModel('gemini-pro')

def send_message(message, history):
    history.append({"role":"user", "parts":message})
    chat = model.start_chat(history = history)
    response = chat.send_message(message)
    history.append({"role":"model", "parts":response.text})
    return response.text, history

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    history = request.json.get('history')
    response, history = send_message(user_message, history)
    return jsonify({'message': response, "history":history})

@app.route('/process-text', methods=['POST'])
def process_text():
    user_data = request.get_json()
    user_text = user_data.get('text', '')

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Generate audio from text using ElevenLabs
    audio_generator = client.generate(
        text=user_text,
        voice="Brian",
        model="eleven_multilingual_v2"
    )
    
    # Store audio in a BytesIO buffer
    audio_buffer = BytesIO()
    for chunk in audio_generator:
        audio_buffer.write(chunk)

    # Rewind the buffer to the beginning
    audio_buffer.seek(0)

    # Send the audio as a file-like response
    return send_file(audio_buffer, mimetype='audio/mpeg', as_attachment=False, download_name="response.mp3")

if __name__ == '__main__':
    app.run(debug=True)
