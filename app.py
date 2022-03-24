from flask import Flask,request, render_template
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model=load_model('Digit Recognisation')

app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
# saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
def predict_digit(img):
    # img_flattern=x_train.reshape(1,28*28)
    predict=model.predict(img)
    final_digit=np.argmax(predict)
    return final_digit

def prediction(img_path):
    i=image.load_img(img_path,target_size=(28,28))
    i=i.convert('L')
    i=image.img_to_array(i)/255.0
    print(i.size)
    i=i.reshape(1,28,28)
    p=predict_digit(i)
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