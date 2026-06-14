from flask import Flask, jsonify, request, Response
import json

app = Flask(__name__)


@app.route("/")
def home():
    return "<h1>Welcome to the lab</h1>"

if __name__ == "__main__":
    app.run(host="127.0.0.1")