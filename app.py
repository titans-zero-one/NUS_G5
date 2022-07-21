from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
import statistics

from PIL import Image
import numpy as np
from skimage import transform

app = Flask(__name__)

model = keras.models.load_model(r'assets\second.h5')
labels = ['Fresh','Fresh','Fresh','Rotten','Rotten','Rotten']

@app.route('/')
def index():
	return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
	IMG_SHAPE = 150
	img = request.files['img']
	img.save("img.jpg")

	model1 = tf.keras.models.load_model(r'./assets/first.h5')

	def load(filename,shape=150):
		np_image = Image.open(filename)
		np_image = np.array(np_image).astype('float32')/255
		np_image = transform.resize(np_image, (shape, shape, 3))
		np_image = np.expand_dims(np_image, axis=0)
		return np_image
	
	pred = labels[np.argmax(model1.predict(load('img.jpg',200)))]
	print(pred)

	return render_template("prediction.html", data=pred)

if __name__ == "__main__":
	app.run(debug=True)