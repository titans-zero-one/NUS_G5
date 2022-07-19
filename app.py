from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

model = keras.models.load_model(r'.\first.h5')
labels = ['fapples','fbanana','foranges','rapples','rbanana','roranges']

cap = cv2.VideoCapture(0)	

# while True:
# 	ret,frame = cap.read()
# 	imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

@app.route('/')
def index():
	return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']
	img.save("img.jpg")
	image = cv2.imread("img.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (200,200))
	image = np.reshape(image, (1,200,200,3))
	pred = model.predict(image)
	pred = np.argmax(pred)
	pred = labels[pred]

	return render_template("prediction.html", data=pred)

if __name__ == "__main__":
	app.run(debug=True)