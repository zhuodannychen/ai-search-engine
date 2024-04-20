from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the simple Flask API!"

@app.route('/data', methods=['GET'])
def get_data():
    sample_data = {
        "id": 1,
        "name": "John Doe",
        "email": "johndoe@example.com"
    }
    return jsonify(sample_data)

if __name__ == '__main__':
    app.run(debug=True)
