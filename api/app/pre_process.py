import os
import random
import json

from sklearn.model_selection import train_test_split
from collections import Counter

from wiki_utils import *



tags = ['positive', 'negative']



data = []
counter = 0

for tag in tags:
    
    folder = './dataset/training/' + tag
    files = os.listdir(folder)
    
    for file in files:
        
        name = folder + '/' + file

        try:
            item = get_attrbs(name, tag, 1)
            data.append(item)

        except:
            pass
        
        counter += 1
        
        print('No. of data points added : {}'.format(counter), end = '\r')


# Train-Validation-Test Split
random.shuffle(data)
train_prop = 0.8



labels = [itm['label'] for itm in data]
train, test = train_test_split(data, test_size = 1-train_prop, stratify = labels)

labels = [itm['label'] for itm in test]
val, test = train_test_split(test, test_size = 0.5, stratify = labels)


train_labels = [itm['label'] for itm in train]
val_labels = [itm['label'] for itm in val]
test_labels = [itm['label'] for itm in test]


print('Percent of positive class in train_set : {:.2f}'.format(Counter(train_labels)['positive']*100/len(train_labels)))
print('Percent of positive class in val_set : {:.2f}'.format(Counter(val_labels)['positive']*100/len(val_labels)))
print('Percent of positive class in test_set : {:.2f}'.format(Counter(test_labels)['positive']*100/len(test_labels)))


with open('./cleaned_datasets/intro_1_para/train.txt', 'w') as stream:
    json.dump({'data':train}, stream)

with open('./cleaned_datasets/intro_1_para/val.txt', 'w') as stream:
    json.dump({'data':val}, stream)

with open('./cleaned_datasets/intro_1_para/test.txt', 'w') as stream:
    json.dump({'data':test}, stream)