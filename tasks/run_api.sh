#!/bin/bash
cd ./api
docker run -d --name wiki_classifier_api -p 5000:80 wiki_classifier:1.0