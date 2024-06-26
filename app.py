from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect
from PIL import Image
import os, io, sys
import numpy as np
import cv2
import base64

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

import simpleaudio as sa

# Load the .wav file
wave_obj = sa.WaveObject.from_wave_file("./alarm.wav")



# Mengubah Path agar kompatibel dengan Windows
# pathlib.PosixPath = pathlib.WindowsPath

# Path ke model YOLOv5 yang sudah dilatih
model_path = './yolov5/runs/exp/weights/best (2).pt'

# Memuat model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Menggunakan kamera video (0 menunjukkan kamera default)
cap = cv2.VideoCapture(0)

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


def gen_frames():
    camera = cv2.VideoCapture(0)  # Mengakses kamera
    # Loop untuk membaca setiap frame dari kamera
    while camera.isOpened():
        # Membaca frame dari kamera
        ret, frame = camera.read()
        
        if not ret:
            break
        
        # Mendapatkan hasil deteksi objek dari model YOLOv5
        results = model(frame)

        # Memeriksa hasil deteksi
        labels = results.pandas().xyxy[0]['name'].tolist()
        current_time = time.time()

        # Update counts dan kondisi untuk setiap kelas
        for label in prediction_counts.keys():
            if label in labels:
                prediction_counts[label].append(current_time)
                if len(prediction_counts[label]) >= 6 and (current_time - prediction_counts[label][0] <= time_limits[label]):
                    # Play the audio
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
            else:
                # Hapus waktu yang sudah terlalu lama
                while prediction_counts[label] and (current_time - prediction_counts[label][0] > time_limits[label]):
                    prediction_counts[label].popleft()

                    # Menggunakan generator untuk stream frame
        
        # Menggambar bounding boxes pada frame
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        result_arr = results.render()[0]
        # Encode frame ke format JPEG
        ret, buffer = cv2.imencode('.jpg', result_arr)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

       
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

# Route untuk menjalankan skrip deteksi kamera
@app.route("/opencam", methods=['GET'])
def opencam():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=False, threaded=False)   #ganti dengan alamat ipmu
