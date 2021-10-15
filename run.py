# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:31:02 2021

@author: Prateek Prabhakar
"""
from TrainingAndPrediction import TrainingAndPrediction
from werkzeug import secure_filename
from pathlib import Path
import os
from flask import Flask, request, session, jsonify
app = Flask(__name__)


@app.route('/')
def test():
    return 'Api is running!'

@app.route('/screen', methods=['POST'])
def screenResume():
    file = request.files['file']
    filePath = save_to_local(file)
    csv_path = os.path.join('./Training','ResumeData.csv')
    pred = TrainingAndPrediction().train_model_and_predict(csv_path, filePath, session)
    return jsonify({"Predicted Class": str(pred)})

def save_to_local(file):
    Path("./Input").mkdir(parents=True, exist_ok=True)
    filePath = os.path.join('./Input',secure_filename(file.filename))
    file.save(filePath)
    return filePath


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    
    app.run()