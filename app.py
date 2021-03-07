from flask import Flask,request, jsonify

app = Flask(__name__)

@app.route("/webhook", methods=['POST'])
def index():
    print(request.get_json())
    return "received from RASA server"

if __name__ == "__main__":
    app.run(debug=True)
