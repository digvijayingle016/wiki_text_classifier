# The Problem: Classify Wikipedia Disease Articles
This repository provides a Deep Learning based solution to classify Disease Articles from Wikipedia. It includes python scripts to pre-process html documents, train new model by tweaking model hyperparameters or use existing model for inference. <br/>

It also includes model weights trained based on 13693 html documents (3693 positive + 10000 negative). The inputs to the model include:
*  **Title** : Title of the wikipedia page extracted using <title></title> tags
*  **Table of Contents**: List of section headers from table of contents section of the wikipedia page
*  **Introduction**: First paragraph from the Introduction section of the wikipedia page

**Performance Metrics**:

|  Type | Accuracy |
| :------------- | :-------------: |
| Train | 99.5% |
| Validation  | 97.66% |
| Test  | 98.46%  |

<br/>

## Instructions To Run The Solution
The user is required to clone this repository and spin-up the docker container using following commands. Once the docker container is spawn, a flask app is launched which can receive requests and produce results based on it.<br />

#### Build Docker Image:
In repo root:
`sh ./tasks/build_api_docker.sh`

#### Spin-up Docker Container:
In repo root:
`sh ./tasks/run_api.sh`

#### Classify Given html Files Into Positve/Negative Class
The user needs to add html files corresponding to wikipedia articles to the test cases directory (api/test_cases). The files from this directory are read using the following commands and a request is posted to Flask app running inside the docker container to produce positve/negative flags for the articles.

In repo root:
`sh ./tasks/test_api.sh`


#### Stop Docker Container
In repo root:
`sh ./tasks/stop_api.sh`

<br/>

## Instructions To Train a New Model
1. Add Json files for train, validation and test datasets to api/app/cleaned_datasets directory. The structure of the json file should be as described in wiki_dataset.py file in api/app folder. Alternatively, create new directory - 'training'(containing sub-directories - 'positive' and 'negative' consisting of respective html documents) in api/app/dataset directory and run pre_process.py file.

2. Update the params.yaml file in api/app/model directory to tweak the model architecture. This step can be skipped in order to continue with existing architecture

3. Run the following command in terminal with appropriate batch_size, n_epochs and model_file_name
`python train.py './model/params.yaml' batch_size n_epochs model_file_name`
