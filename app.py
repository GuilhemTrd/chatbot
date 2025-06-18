from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/')
def home():
    return "Welcome to the Chatbot API!"
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = f"You said:{user_message}"
    return jsonify({"response": response})
if __name__ == '__main__':
    app.run(debug=True)