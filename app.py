from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

model = keras.models.load_model(r'assets\third_veg.h5')
labels = ['Fresh Apples','Fresh Banana','Fresh Oranges','Rotten Apples','Rotten Banana','Rotten Oranges']

@app.route('/')
def index():
	return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
	IMG_SHAPE = 150
	img = request.files['img']
	img.save("img.jpg")
	image = cv2.imread("img.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (IMG_SHAPE,IMG_SHAPE))
	image = np.reshape(image, (1,IMG_SHAPE,IMG_SHAPE,3))
	pred = model.predict(image)
	pred = np.argmax(pred)
	print(pred)
	pred = labels[pred]

	return render_template("prediction.html", data=pred)

if __name__ == "__main__":
	app.run(debug=True)