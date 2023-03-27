import numpy as np
import pandas as pd
from fastai.text.all import *
import torch

payloads = [
    {'id':1, 'text':"today is a good day"}
    ]

# load model
learner = load_learner(r"..\..\model\classifier_model.pkl", cpu=False)

# Read the classifier data
fastai_text_classifier_data = torch.load(r"..\..\classifier_data.pkl")

# retain the class order
classes = fastai_text_classifier_data.categorize.vocab

def make_prediction(payload, learner, classes):
    text = payload['text']
    dl = learner.dls.test_dl(text)
    proba = learner.get_preds(dl=dl)
    pred_index = np.argmax(proba[0][0], axis=0)
    payload['predicted_class'] = classes[pred_index]
    payload['probability'] = proba[0][0][pred_index]
    print(payload)
    return payload

if __name__ == '__main__':
    for payload in payloads:
        make_prediction(payload, learner, classes)