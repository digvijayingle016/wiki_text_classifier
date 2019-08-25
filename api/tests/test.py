import os
import sys
import requests
import json

from pathlib import Path



def read_xml(filepath):
    '''
    Documentation : TODO
    '''
    content = open(filepath, 'r').read()
    file = filepath.split('/')[-1]

    return file, content


def post(files, test_dir, uri='localhost', port=5000, method='wiki_classifier/predict', version='v1'):
    
    if len(files) == 0:
        return 'Empty TEST_CASES_DIR. Please add files to the directory'
    

    headers = {'Content-type': 'application/json'}

    
    data = []

    for file in files:
        name, content = read_xml(str(test_dir) + '/' + file)
        data.append({'id': name, 'content': content})


    endpoint = 'http://{}:{}/{}/{}'.format(uri, port, method, version)
    
    r = requests.post(endpoint, data=json.dumps({'data': data}), headers=headers)

    if r.ok:
        if r.json():
            results = r.json()
            return results
        else:
            return('Empty return. Wiki Classifier Model could not recognize the as reported terms')
    else:
        return 'Status code: {}'.format(r.status_code)


uri = 'localhost'
port = 5000
method = 'wiki_classifier/predict'
version = 'v1'


TEST_CASES_DIR = Path(sys.path[0]).parents[0] / 'test_cases'


headers = {'Content-type': 'application/json'}


files = os.listdir(TEST_CASES_DIR)


json_response = post(files, TEST_CASES_DIR, uri, port, method, version)


print(json_response)