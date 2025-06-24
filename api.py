from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chat_NLP import ChatbotNLP
import os

app = Flask(__name__)
CORS(app)
bot = ChatbotNLP()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    intent = data.get("intent")
    message = data.get("message")

    if intent:
        response = bot.handle_intention(intent.upper())
        return jsonify({"response": response})

    if message:
        response = bot.process_input(message)
        return jsonify({"response": response})

    return jsonify({"response": "Je n'ai pas compris votre demande."}), 400

@app.route("/")
def serve_index():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(os.getcwd(), path)

if __name__ == "__main__":
    # Récupérer le port depuis l'environnement (Render/Vercel/OVH), fallback 5000 pour local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
