from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    """Return the client application."""
    return render_template("audio/main.html")