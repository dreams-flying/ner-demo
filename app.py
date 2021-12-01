#! -*- coding:utf-8 -*-
import keras
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from utils.ner_predict import model
from utils.ner_predict import extract_arguments

sess=keras.backend.get_session()
graph=tf.get_default_graph()
model.load_weights('./utils/save/save_albert/best_model.weights')#
app = Flask(__name__)

@app.route('/', methods=['GET',])
def index():
    return render_template('index.html')

@app.route('/nerapi', methods=['POST',])
def nerapi():
    if request.method == 'GET':
        return "<h1>error</h1>"
    else:
        text = request.form.get('sentences')
        # 此处写flask调用keras的代码逻辑
        with sess.as_default():
            with graph.as_default():
                result = extract_arguments(text)
    return render_template('index.html', result = zip(result[0],list(text)))


if __name__ == '__main__':
    app.debug = True
    app.run()#
