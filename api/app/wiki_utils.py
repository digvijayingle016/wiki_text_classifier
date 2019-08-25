import re
import time
import gensim
import pickle
import yaml
import numpy as np

import torch

from classifier import LSTMClassifier
from bs4 import BeautifulSoup


class DataIterator(object):
    '''
    Creates an iterator for passing data in batches during inference, especially when large number of requests are posted
    
    Arguments:
    data : data to create batches from 
    batch_size: integer specifying size of each batch
    '''
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.iter = self.make_random_iter()
        
    def next_batch(self):
        '''
        Returns next batch
        '''
        try:
            idxs = self.iter.__next__()
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = self.iter.__next__()
        titles = [self.data[i]['title'] for i in idxs]

        tocs = []
        for i in idxs:
            toc = self.data[i]['toc']
            if toc == []:
                tocs.append('.')
            else:
                tocs.append(' '.join(toc))

        intros = [self.data[i]['intro'] for i in idxs]
        
        cas_tags = [self.data[i]['cas_tag'] for i in idxs]
        
        return titles, tocs, intros, cas_tags

    def make_random_iter(self):
        '''
        Returns an iterator for batches
        '''
        splits = np.arange(self.batch_size, len(self.data)+self.batch_size, self.batch_size)
        self.batch_count = len(splits)
        it = np.split(range(len(self.data)), splits)[:-1]
        return iter(it)


def load_model(model_file_name, params_file):
    '''
    Initializes a model object and loads trained weights from given file name

    Arguments:
    model_file_name: string specifying the location of trained model weights
    params_file: string specifying location to yaml file containing hyperparameters required to initialize the model object

    Returns
    classifier: model object with trained model weights loaded
    '''

    state_dict  = torch.load(model_file_name, map_location=torch.device('cpu'))

    with open(params_file, 'r') as stream:
        params = yaml.load(stream)

    classifier = create_model(params)

    classifier.load_state_dict(state_dict)
    
    return classifier


def create_model(params):
    '''
    Creates a model objects using the hyperparameters from the dictionay 'params'

    Arguments:
    params: a dictionary containing the hyperparameters required to create the model object

    Returns:
    model: model object
    '''
    emb_size = params['emb_size']
    hidden_size = params['hidden_size']
    n_lstm_layers = params['n_lstm_layers']
    num_classes = params['num_classes']
    bidirectional = params['bidirectional']
    dense_layer_sizes = params['dense_layer_sizes']
    dropout_prob = params['dropout_prob']

    model = LSTMClassifier(emb_size = emb_size,
                           hidden_size = hidden_size,
                           n_lstm_layers = n_lstm_layers,
                           num_classes = num_classes,
                           bidirectional = bidirectional,
                           dense_layer_sizes = dense_layer_sizes,
                           dropout_prob = dropout_prob)
    return model


def load_w2v_model(path):
    '''
    Loads the word2vec model using path
    
    Arguments:
        path : string specifying the location of the word2vec model
        
    Returns: 
        wordsModel: Word2Vec model
    '''
    wordsModel = gensim.models.Word2Vec.load(path)
    return wordsModel



def create_embedding_matrix(wordsModel, dataset_obj):
    '''
    Creates a 2D tensor by stacking word vectors from wordsModel in the order of words in the vocabulary of the dataset_obj. This 2D tensor is passed to the '_weights' argument of the embeddings layer.    

    Arguments:
    wordsModel: word2vec model
    dataset_obj: A Pytorch Dataset object

    Return:
    embedding_matrix: A 2D tensor of shape (vocab_size X embedding_dimension)
    '''
    emb_mat = torch.zeros(dataset_obj.text_encoder.vocab_size, wordsModel.vector_size)
    
    for i in range(emb_mat.size(0)):
        try:
            emb_mat[i,:] = torch.tensor(wordsModel.wv.__getitem__(dataset_obj.text_encoder.decode(torch.tensor(i))))
        except:
            pass
    return emb_mat


def save_model(model, file_name):
    '''
    Saves state_dict of the model

    Arguments:
    model: Model object
    file_name:  
    '''
    torch.save(model.state_dict(), file_name)


def train_model(model, epochs, trainset, valset, criterion, optimizer, device, file_name):
    '''
    Performs the following tasks

    1) Performs forward propagation, calculates loss and carries out backpropagation using the specified criterion and optimizer
    2) Evaluates the model performance on training and validation set
    3) Carries out model checkpointing if the validation accuracy improves in current epoch


    Arguments: 
    model: model object
    epochs: No. of epochs
    trainset: Dataset object corresponding to training data
    valset: Dataset object corresponding to validation data
    criterion: Criterion to calculate the loss 
    optimizer: Optimizer to minimize the loss
    device: Specifies the device to be used for storing the tensors and carrying our computations
    file_name: File name to save the model's state_dict into

    Returns: 
    model: A model object with trained weights
    '''

    #Lists to store Loss and Accuracy:
    train_loss_ = []
    train_acc_ = []

    val_loss_ = []
    val_acc_ = []

    start_time = time.time()

    with torch.set_grad_enabled(True):

        best_val_acc = 0

        for epoch in range(epochs):

            model.train()

            train_acc = 0.0
            train_loss = 0.0
            total = 0.0

            for title_attrs, toc_attrs, intro_attrs, labels in trainset:

                torch.cuda.empty_cache()

                title_attrs = (title_attrs[0].to(device), title_attrs[1].to(device))
                toc_attrs   = (toc_attrs[0].to(device), toc_attrs[1].to(device))
                intro_attrs = (intro_attrs[0].to(device), intro_attrs[1].to(device))

                labels = labels.to(device)
                
                optimizer.zero_grad()
                output = model.forward(title_attrs, toc_attrs, intro_attrs)
                loss = criterion(output, labels)

                loss.backward()

                optimizer.step()

                _, predicted_index = torch.max(output,  1)

                # Calculate total loss and accurate predictions till current batch
                total += len(labels)
                train_loss += loss.item()
                train_acc += (predicted_index == labels).sum().item()

            # Store logs for training loss and accuracy for current batch
            train_loss_.append(train_loss/total)
            train_acc_.append(train_acc/total)

            # Evaluate performance on validation set and save the corresponding logs
            val_loss, val_acc = calculate_score(model, valset, criterion, device)

            val_loss_.append(val_loss.item())
            val_acc_.append(val_acc)

            print('Epoch %3d/%3d : Training Loss : %.5f, Training Accuracy : %.4f, Validation Loss : %.5f, Validation Accuracy : %.4f, Time : %.2f' % (epoch+1, epochs, train_loss/total, train_acc/total, val_loss, val_acc, time.time()-start_time))

            # Save current model weights in current validation accuracy > last best validation accuracy 
            if val_acc >= best_val_acc:
                save_model(model, file_name + str('.pt'))
                best_val_acc = val_acc

    logs = {'train_loss': train_loss_, 
        'train_acc': train_acc_,
        'val_loss': val_loss_,
        'val_acc': val_acc_}

    # Save the logs file for visualizing learning in future
    with open(file_name + '_logs.pkl', 'wb') as file:
        pickle.dump(logs, file)        

    return model


def calculate_score(model, dataset, criterion, device):

    '''
    Evaluates the model on validation set

    Arguments: 
    model: A model object
    dataset: Dataset object corresponding to validation data
    criterion: Criterion to calculate validation loss
    device: Specifies the device to be used for storing the tensors and carrying our computations

    Returns:
    output: softmax output
    loss: Loss
    acc: Accuracy
    '''

    loss = 0.0
    acc = 0.0
    total = 0.0

    model.eval()

    with torch.set_grad_enabled(False):

        for title_attrs, toc_attrs, intro_attrs, labels in dataset:

            torch.cuda.empty_cache()

            title_attrs = (title_attrs[0].to(device), title_attrs[1].to(device))
            toc_attrs   = (toc_attrs[0].to(device), toc_attrs[1].to(device))
            intro_attrs = (intro_attrs[0].to(device), intro_attrs[1].to(device))

            labels = labels.to(device)

            output = model.forward(title_attrs, toc_attrs, intro_attrs)

            loss = criterion(output, labels)

            _, predicted_index = torch.max(output, 1)

            # Calculate total loss and accurate predictions till current batch 
            total += len(labels)
            loss += loss.item()
            acc += (predicted_index == labels).sum().item()

    return loss/total, acc/total


def clean_text(text):
    '''
    Removes non alpha-numeric characters, converts text to lower case and removes extra spaces

    Arguments:
    text: text to be cleaned

    Returns:
    cleaned_text
    '''
    cleaned_text = ' '.join(re.sub('[^a-zA-Z0-9]', ' ', text).lower().strip().split())
    
    return cleaned_text


def extract_table_of_contents(soup):
    '''
    Extracts table of contents from the toc section of wiki page

    Arguments:
    soup: a BeautifulSoup object corresponding to the xml file of the wiki page

    Returns:
    cleaned_contents: list of items in the toc section of the wiki page
    '''
    contents = []
    
    for sub_elem in soup.find_all('li'):
        
        attrs = list(sub_elem.attrs.keys())
        
        if 'class' in attrs:            
            
            if sub_elem.attrs['class'].find('toclevel') >= 0:
                text = sub_elem.contents[0].text
                contents.append(re.sub('^\d+(\.\d+)*\s','', text))
    
    cleaned_contents = [clean_text(content) for content in contents]
    
    return cleaned_contents


def extract_intro(soup, n = 2):
    '''
    Extracts Introduction section(based on first n <p> tags) from a wiki page. In total number of <p></p> tags < n, it sequentially combines the texts 
    enclosed in <p></p> tags until the total number of tokens is atleast 5.

    Arguments:
    soup: a BeautifulSoup object corresponding to the xml file of the wiki page
    n: number of <p></p> tags to extract the text from 

    Returns:
    cleaned version of introduction of wiki page
    '''
    ps = soup.find_all('p')
    
    intro = ' '.join([ps[idx].text for idx in range(min(len(ps),n))])
    
    if len(intro.split()) < 5:
        
        ln = 0
        
        while ln < 5:
            
            n += 1
            intro = ' '.join([ps[idx].text for idx in range(min(len(ps),n))])
            ln = len(intro.split())
    
    return clean_text(intro)


def extract_title(soup):
    '''
    Extracts Title of the wiki page

    Arguements:
    soup: a BeautifulSoup object corresponding to the xml file of the wiki page

    Returns:
    cleaned version of title of the wiki page
    '''
    title = soup.find('title').get_text()
    
    return clean_text(title.split('-')[0].strip())


def get_attrbs(filename, tag, n_paras = 1):
    '''
    Extracts title, table of contents, introduction tag from the xml file and assigns label to it

    Arguments: 
    filename: location of file
    tag: 'positive' if file corresponds to disease article else 'negative'
    n_paras: default = 1, number of paragraphs to extract from the introduction of wiki page

    Returns:
    Dictionary containing following attributes of the wiki page

    {'id': filename,
     'title': title of page,
     'toc': list of items in table of contents section of wikipage,
     'intro': introduction of wiki page,
     'label':'positive'/'negative' flag}
    '''
    content = open(filename, 'r').read()
    soup = BeautifulSoup(content, 'xml')    
    
    title = extract_title(soup)
    toc = extract_table_of_contents(soup)
    intro = extract_intro(soup, n_paras)
    label = tag
    
    file = filename.split('/')[-1]

    item = {'id' : file,
            'title' : title,
            'toc' : toc,
            'intro': intro,
            'label': label}
    
    return item


def clean_xml(file, content, n_paras = 1):
    '''
    Extracts title, table of contents, introduction tag from the xml file and checks is CAS Registry Number is present in wiki page. This function is used during 
    inference to extract relevant attributes and check for drug related articles

    Arguments:
    file: filename
    content: xml_doc
    n_paras: default = 1, number of paragraphs to extract from the introduction of wiki page

    Returns: 
    Dictionary containing following attributes

    {'id': filename,
     'title': title of page,
     'toc': list of items in table of contents section of wikipage,
     'intro': introduction of wiki page,
     'cas_tag':0/1 flag indicating presence of CAS Registry Number}
    '''
    soup = BeautifulSoup(content, 'xml')    
    
    title = extract_title(soup)
    toc = extract_table_of_contents(soup)
    intro = extract_intro(soup, n_paras)
    
    cas_tag = int(str(soup).find('CAS Number') != -1)
    
    item = {'id' : file,
            'title' : title,
            'toc' : toc,
            'intro': intro,
            'cas_tag': cas_tag}
    
    return item


def read_xml(filepath):
    '''
    Reads the xml file

    Arguments:
    filepath: location of file

    Return:
    file: filename
    content: xml doc
    '''
    content = open(filepath, 'r').read()
    file = filepath.split('/')[-1]

    return file, content
