from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect
from PIL import Image
import os, io, sys
import numpy as np
import cv2
import base64
from io import BytesIO
from werkzeug.utils import secure_filename, send_from_directory
from yolo_detection import run_model
from language_conversion import convert_lang
import subprocess

# from pathlib import Path
# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath

from flask import Flask, Response
import cv2

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from collections import deque

from pydub import AudioSegment
from pydub.playback import play
import threading

# Load the .wav file
x, w, h, y =  0,0,0,0


def play_audio(file_path):
    audio = AudioSegment.from_mp3(file_path)
    play(audio)

# Mengubah Path agar kompatibel dengan Windows
# pathlib.PosixPath = pathlib.WindowsPath

# Path ke model YOLOv5 yang sudah dilatih
model_path = './yolov5/runs/exp/weights/best.pt'

# Memuat model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Variabel untuk menyimpan prediksi dalam periode waktu tertentu
app = Flask(__name__)

prediction_counts = {
    'DangerousDriving': deque(maxlen=6),
    'Distracted': deque(maxlen=6),
    'Drinking': deque(maxlen=6),
    'SleepyDriving': deque(maxlen=6),
    'Yawn': deque(maxlen=6)
}

# Jangka waktu deteksi dalam detik
time_limits = {
    'DangerousDriving': 2,
    'Distracted': 10,
    'Drinking': 4,
    'SleepyDriving': 5,
    'Yawn': 6
}

# Route untuk memproses gambar, menjalankan model deteksi objek, dan konversi bahasa
@app.route('/project_massive', methods=['POST'])
def mask_image():
    # Membaca file gambar dari request
    file = request.files['image'].read() 
    npimg = np.frombuffer(file,np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)[:, :, ::-1]
    # Menjalankan model deteksi objek
    img,text = run_model(img)
    print("{} This is from app.py".format(text))
    if(text.lower() == "he is"):
        text = ""
    # Menyusun teks dalam bahasa Inggris dan Indonesia
    if(len(text) == 0):
        text = "Reload the page and try with another better image"
    
    englishtext = text
    indotext = convert_lang(text)
    # Mengonversi gambar ke format base64 untuk dikirim sebagai respons JSON
    bufferedBytes = io.BytesIO()
    img_base64 = Image.fromarray(img)
    img_base64.save(bufferedBytes, format="JPEG")
    img_base64 = base64.b64encode(bufferedBytes.getvalue())
    
    return jsonify({'status':str(img_base64),'englishmessage':englishtext, 'indomessage':indotext})


# Route untuk uji coba
@app.route('/test', methods=['GET', 'POST'])
def test():
	print("log: got at test", file=sys.stderr)
	return jsonify({'status': 'succces'}) 

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Fungsi untuk menambahkan header CORS setiap kali respons dikirim
@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers','Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

@app.route('/detect', methods=['GET'])
def detect():
    # threading.Thread(target=play_audio, args=('./alarm.wav',)).start()
    return render_template('detect.html')

# Route untuk menjalankan skrip deteksi kamera
@app.route("/opencam", methods=['POST'])
def opencam():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," part
    image = Image.open(BytesIO(base64.b64decode(image_data)))


       # Mendapatkan hasil deteksi objek dari model YOLOv5
    result = model(image)
    

    
    
    if (result.pandas().xywh[0].shape[0] == 0 ): return {
        "objects": [
             {"x": 0, "y": 0, "width": 0, "height": 0}
        ]
    }
    
 
    
    
    w = result.pandas().xywh[0]["width"][0]
    h = result.pandas().xywh[0]["height"][0]
    x = result.pandas().xywh[0]["xcenter"][0] - (w//2)
    y= result.pandas().xywh[0]["ycenter"][0] - (h//2)
    

    confidence = round(float(result.crop()[0]["conf"]),2)
 
    # Memeriksa hasil deteksi
    current_time = time.time()

    danger=0
    print(prediction_counts)
    # Update counts dan kondisi untuk setiap kelas
    labels = result.pandas().xyxy[0]['name'].tolist()
    for label in prediction_counts.keys():
        if label in labels:
            prediction_counts[label].append(current_time)
            if len(prediction_counts[label]) >= 6 and (current_time - prediction_counts[label][0] <= time_limits[label]):
                # Play the audio
                print("DETECTED &*&**^*&^*&^&^&*^&*^")
                print("DETECTED &*&**^*&^*&^&^&*^&*^")
                print("DETECTED &*&**^*&^*&^&^&*^&*^")
                danger=1

        else:
            # Hapus waktu yang sudah terlalu lama
            while prediction_counts[label] and (current_time - prediction_counts[label][0] > time_limits[label]):
                prediction_counts[label].popleft()

                # Menggunakan generator untuk stream frame
    label = result.pandas().xyxy[0]['name'][0]
    response = {
        "objects": [
             {"x": x, "y": y, "width": w, "height": h, "label": label, "confidence": confidence, "danger":danger}
        ]
    }
                
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=True, threaded=False)   #ganti dengan alamat ipmu

