from flask import Flask,request, url_for, redirect, render_template
import os
import flask
import pickle
import matplotlib.image as img
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

import digit_recognition as dj

app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
# saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)

def prediction(img_path):
    i=image.load_img(img_path,target_size=(28,28))
    i=i.convert('L')
    i=image.img_to_array(i)/255.0
    print(i.size)
    i=i.reshape(1,28,28)
    p=dj.predict_digit(i)
    return p

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/image',methods=['POST'])
def predict():
    get_img=request.files['take_image']
    get_img_path="static/" + get_img.filename
    get_img.save(get_img_path)
    p=prediction(get_img_path)
    os.remove(get_img_path)

    return render_template('index.html',pred=p,bhai="kuch karna hain iska ab?")



if __name__ == '__main__':
    app.run(debug=True)