import plac

from wiki_dataset import WikiDataset
from classifier import LSTMClassifier
from wiki_utils import *

import torch
import datetime
import yaml
from sklearn.externals import joblib

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from torchnlp.samplers import BucketBatchSampler


@plac.annotations(
    params_file = ('.yaml file containing the model hyperparameters', 'positional', None, str),
    batch_size = ('no. of training examples to pass in a batch', 'positional', None, int),
    epochs = ('no. of epochs', 'positional', None, int),
    model_file_name = ('file name to save the model weights', 'positional', None, str),
    learning_rate = ('learning_rate', 'option', None, float),
    weight_decay = ('coefficient for weight regularization', 'option', None, float),
    n_workers = ('no. of n_workers', 'option', None, int),
    use_pretrained_embs = ('Use Pretrained Embeddings', 'option', None, bool)
    )


def main(params_file, batch_size, epochs, model_file_name, learning_rate = 1e-3, weight_decay = 1e-5, n_workers = 6, use_pretrained_embs = False):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Files to Load    
    train_json_file = './cleaned_datasets/intro_1_para/train.txt'
    val_json_file = './cleaned_datasets/intro_1_para/val.txt'
    word2vec_model_path  = '/word2vec_100D.w2v'


    print('[' + str(datetime.datetime.now()) + '] : Reading Files')

    if use_pretrained_embs:
        wordsModel = load_w2v_model(word2vec_model_path)
        vocab = sorted(list(wordsModel.wv.vocab))
    else:
        vocab = None


    print('[' + str(datetime.datetime.now()) + '] : Creating Dataset Objects')
    train_dataset = WikiDataset.fromJsonFile(train_json_file, vocab= vocab, mode = 'train')
    val_dataset = WikiDataset.fromJsonFile(val_json_file, text_encoder=train_dataset.text_encoder, label_encoder=train_dataset.label_encoder, vocab= train_dataset.text_encoder.vocab, mode = 'train')

    trainset = DataLoader(train_dataset, num_workers=n_workers, 
                            batch_sampler=BucketBatchSampler(train_dataset.data['data'], batch_size=batch_size, drop_last=True, sort_key=lambda a:-len(a['intro'].split())),  
                            collate_fn=train_dataset.collate_fn)
    valset = DataLoader(val_dataset, num_workers=n_workers, 
                                batch_sampler=BucketBatchSampler(val_dataset.data['data'], batch_size=batch_size, drop_last=True, sort_key=lambda a:-len(a['intro'].split())),  
                                collate_fn=val_dataset.collate_fn)


    print('[' + str(datetime.datetime.now()) + '] : Reading params_file')
    with open(params_file, 'r') as stream:
        params = yaml.load(stream)

    params['emb_size'] = (train_dataset.vocab_size, 100)
    params['num_classes'] = train_dataset.label_encoder.vocab_size


    print('[' + str(datetime.datetime.now()) + '] : Creating Model Object')
    classifier = create_model(params)


    if use_pretrained_embs:     
        print('[' + str(datetime.datetime.now()) + '] : Creating Embedding Matrix')
        embedding_matrix = create_embedding_matrix(wordsModel, train_dataset)

        classifier.embeddings.weight = nn.Parameter(embedding_matrix)

        del embedding_matrix

    
    classifier.to(device)

    
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([0,1.36/1, 1.36/0.36]).to(device))
    optimizer = optim.Adam(classifier.parameters(), lr = learning_rate , weight_decay=weight_decay)

    print('[' + str(datetime.datetime.now()) + '] : Training Model ...')
    classifier = train_model(classifier, epochs, trainset, valset, criterion, optimizer, device, model_file_name)


    model_utils = {'text_encoder': train_dataset.text_encoder,
                  'label_encoder': train_dataset.label_encoder}


    joblib.dump(model_utils, model_file_name + str('_model_utils.pkl'))

    with open(params_file, 'w') as stream:
        yaml.dump(params, stream)

if __name__ == '__main__':
    plac.call(main)