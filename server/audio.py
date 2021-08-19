"""Audio Recording Socket.IO Example
Implements server-side audio recording.
"""
from flask import Flask, current_app, session, render_template
from flask_socketio import emit, SocketIO

import numpy as np
import librosa
import soundfile as sf

import torch
import torch.nn.functional as F

from utils.model import CNN
from utils.transformers import audio_transform, ZmuvTransform

app = Flask(__name__)
app.config["FILEDIR"] = "static/_files/"

socketio = SocketIO(app, logger=True, engineio_logger=True)

wake_words = ["hey", "fourth", "brain"]
classes = wake_words[:]
classes.append("oov")

window_size_ms = 750
# 16 bit signed int. 2^15-1
audio_float_size = 32767
sample_rate = 16000
max_length = int(window_size_ms / 1000 * sample_rate)
no_of_frames = 4

# init model
num_labels = len(wake_words) + 1  # oov
num_maps1 = 48
num_maps2 = 64
num_hidden_input = 768
hidden_size = 128
model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# load trained model
model.load_state_dict(torch.load("trained_models/model_hey_fourth_brain.pt", map_location=torch.device("cpu")))

# load zmuv
zmuv_transform = ZmuvTransform().to(device)
zmuv_transform.load_state_dict(torch.load(str("trained_models/zmuv.pt.bin"), map_location=torch.device("cpu")))


@app.route("/")
def index():
    """Return the client application."""
    return render_template("audio/main.html")


@socketio.on("start-recording", namespace="/audio")
def start_recording(options):
    """Start recording audio from the client."""
    session["bufferSize"] = options.get("bufferSize", 1024)
    session["fps"] = options.get("fps", 44100)

    session["windowSize"] = 0
    session["frames"] = []
    session["batch"] = []
    session["target_state"] = 0
    session["infer_track"] = []


@socketio.on("write-audio", namespace="/audio")
def write_audio(data):
    """Write a chunk of audio from the client."""
    session["frames"].append(data)
    session["windowSize"] += 1
    print(f'{session["windowSize"]} - {int(session["fps"] / session["bufferSize"] * window_size_ms / 1000)}')
    # if we got 750 ms then process
    if session["windowSize"] >= int(session["fps"] / session["bufferSize"] * window_size_ms / 1000):
        # convert stream to numpy
        stream_bytes = [str.encode(i) if type(i) == str else i for i in session["frames"]]
        audio_data = np.frombuffer(b"".join(stream_bytes), dtype=np.int16).astype(np.float) / audio_float_size
        # convert sample rate to 16K
        audio_data = librosa.resample(audio_data, session["fps"], sample_rate)
        print(audio_data.size)
        # for testing write to file
        # sf.write(f'{current_app.config["FILEDIR"]}temp.wav', audio_data, sample_rate)
        audio_data_length = audio_data.size / sample_rate * 1000
        # if given audio is less than window size, pad it
        if audio_data_length < window_size_ms:
            audio_data = np.append(audio_data, np.zeros(int(max_length - audio_data.size)))
        else:
            audio_data = audio_data[:max_length]
        # convert to tensor
        inp = torch.from_numpy(audio_data).float().to(device)
        session["batch"].append(inp)
        # reset
        session["windowSize"] = 0
        session["frames"] = []

    if len(session["batch"]) >= no_of_frames:
        audio_tensors = torch.stack(session["batch"])
        session["batch"] = []  # reset batch
        mel_audio_data = audio_transform(audio_tensors, device, sample_rate)
        mel_audio_data = zmuv_transform(mel_audio_data)
        scores = model(mel_audio_data.unsqueeze(0))
        scores = F.softmax(scores, -1).squeeze(1)
        for score in scores:
            preds = score.cpu().detach().numpy()
            preds = preds / preds.sum()
            pred_idx = np.argmax(preds)
            pred_word = classes[pred_idx]
            print(f"predicted label {pred_idx} - {pred_word}")
            label = wake_words[session["target_state"]]
            if pred_word == label:
                session["target_state"] += 1
                session["infer_track"].append(pred_word)
                emit("add-prediction", pred_word)
                if session["infer_track"] == wake_words:
                    emit("add-prediction", f"Wake word {' '.join(session['infer_track'])} detected")
                    # reset
                    session["target_state"] = 0
                    session["infer_track"] = []


@socketio.on("end-recording", namespace="/audio")
def end_recording():
    """Stop recording audio from the client."""
    del session["frames"]
    del session["batch"]


if __name__ == "__main__":
    socketio.run(app)
