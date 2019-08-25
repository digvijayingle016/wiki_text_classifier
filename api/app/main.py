import os
from datetime import datetime

from wiki_utils import *
from bs4 import BeautifulSoup
from sklearn.externals import joblib

import torch.nn.functional as F

from flask import request, jsonify, Flask

from pathlib import Path

DIR_NAME = Path(__file__).parents[0].resolve() / 'model'


model_file_name = 'all_inputs_lc_clf.pt'
model_utils_file = 'all_inputs_lc_clf_model_utils.pkl'
params_file = 'params.yaml'



VERSION = '/v1'
API_PATH = '/wiki_classifier/predict'


start = datetime.now()

model = load_model(DIR_NAME / model_file_name, DIR_NAME / params_file)
model.eval()


# Load models_utils : Dictionary with keys: text_encoder, label_encoder, mapping_dict
model_utils = joblib.load(DIR_NAME / model_utils_file)


print('Model loaded in {}'.format(datetime.now() - start))


app = Flask(__name__)


@app.route(API_PATH + VERSION, methods = ['POST'])
def get_results():

    '''
    Reads the data from requests, checks for some edge cases and finally passes it to the model for inference
    '''
    reqs = request.get_json()

    if reqs:

        contents = reqs['data']

        data = []
        for itm in contents:
            data.append(clean_xml(itm['id'], itm['content']))

        results = predict(model, model_utils, data)

    return jsonify(results)


def predict(model, model_utils, data):
    '''
    Encodes the input from requests and produces inference from the model
    
    Arguments:
    model: model classifier object
    model_utils: a dictionary consisting of text_encoder, label_encoder, mapping_dict
    data: data to be passed to iterator to generate batches
    
    Returns:
    results: final results json format
    '''
        
    iterator = DataIterator(data, 256)
    
    preds = []
    
    for itr in range(iterator.batch_count):
        
        title_batch, toc_batch, intro_batch, cas_tags = iterator.next_batch()
        
        title_attrs = model_utils['text_encoder'].batch_encode(title_batch)
        toc_attrs = model_utils['text_encoder'].batch_encode(toc_batch)
        intro_attrs = model_utils['text_encoder'].batch_encode(intro_batch)
        
        torch.cuda.empty_cache()
        
        batch_results = get_labels(model, title_attrs, toc_attrs, intro_attrs, model_utils['label_encoder'])
        
        batch_results = ['negative' if cas_tags[i] == 1 else batch_results[i] for i in range(len(batch_results))]
        
        preds.extend(batch_results)
    
    results = compile_results(data, preds)

    return {'results': results}


def get_labels(model, title_attrs, toc_attrs, intro_attrs, label_encoder):
    '''
    Carries forward propagation through the model and decodes labels using the label encoder

    Arguments:
    model: model classifier object
    title_attrs: tuple containing padded sequence tensor for title and sequence lengths 
    toc_attrs: tuple containing padded sequence tensor for table of contents and sequence lengths
    intro_attrs: tuple containing padded sequence tensor for introduction and sequence lengths 
    label_encoder: encoder object that encodes labels to their unique integer ids

    Returns:
    labels: list of labels for input data
    '''    
    output = model.forward(title_attrs, toc_attrs, intro_attrs)
    output = F.softmax(output, dim = 1)
    
    _, idx = torch.sort(output, dim = 1, descending = True)
    
    idx = idx[:,0]
    
    labels = label_encoder.batch_decode(idx)
    
    return labels


def compile_results(data, preds):
    '''
    Organizes the results into a json structure

    Arguments:
    data: input data
    preds: predictions from the model

    Returns:
    results: results in json format
            JSON structure:
            [{'id': filename, 'title': title of wiki page, 'labels': predicted label}, 
             {'id': filename, 'title': title of wiki page, 'labels': predicted label}]
    '''
    results = []
    for idx in range(len(data)):
        item = {'file' : data[idx]['id'],
                'title' : data[idx]['title'],
                'labels': preds[idx]}
        results.append(item)

    return results


if __name__ == "__main__":
    app.run(port=5000,debug=False)