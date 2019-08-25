import json
import collections

import torch
from torch.utils.data import Dataset

from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors
from torchnlp.encoders import LabelEncoder


class WikiDataset(Dataset):
    '''
    A custom dataset object that encodes a tokenized text and its labels according to the corresponding encoders 
    '''
    def __init__(self, json, text_encoder = None, label_encoder = None, vocab = None, mode = 'train'):
        '''
        Initialization
        Arguments:
        json: Json file containing the data. 
            Structure of json file:
            e.g: 
                 json: {'data' : [{'id': filename,
                                   'title': title of page,
                                   'toc': [list of items in table of contents section of wikipage],
                                   'intro':introduction of wiki page,
                                   'label':'positive'/'negative' flag}]
                        }
            Labels-required only when mode = 'train'
        text_encoder: encoder object that encodes tokens to their unique integer ids
        label_encoder: encoder object that encodes labels to their unique integer ids
        vocab: external vocabulary used to intialize the text encoder. If vocab = None, it would be generated based on tokens from the datasets provided
        mode: 'train' or 'inference': in case of mode == 'inference', the dataset object skips the labels
        '''
        self.data = json
        assert 'data' in self.data


        # Define the mode in which the dataset object is to be used
        self.mode = mode


        # Define text encoder and vocabulary
        if text_encoder:
            self._text_encoder = text_encoder
            self._vocab = self._text_encoder.vocab
        elif vocab:
            self._vocab = vocab
            self._text_encoder = StaticTokenizerEncoder(self._vocab, append_eos=False, tokenize = self.split)
        else:
            self._vocab = self.create_vocab()   
            self._text_encoder = StaticTokenizerEncoder(self._vocab, append_eos=False, tokenize = self.split)

        self._vocab_size = self._text_encoder.vocab_size


        # Define label encoder
        if self.mode == 'train':
            if label_encoder:
                self._label_encoder = label_encoder
            else:
                self._label_encoder = LabelEncoder([sample['label'] for sample in self.data['data']])

            self._label_size = self._label_encoder.vocab_size

        else:
            self._label_encoder = None
            self._label_size = None


    def __len__(self):
        '''
        Size of dataset
        '''
        return len(self.data['data'])


    def __getitem__(self, idx):
        '''
        Extract item corresponding to idx'th index in data
        '''
        item = self.data['data'][idx]

        intro_enc = self._text_encoder.encode(item['intro'])
        
        toc = item['toc']
        if toc == []:
            toc_enc = self._text_encoder.encode('.')
        else:
            toc = ' '.join(toc)
            toc_enc = self._text_encoder.encode(toc)

        title_enc = self._text_encoder.encode(item['title'])

        if self.mode == 'train':
            return title_enc, toc_enc, intro_enc, self._label_encoder.encode(item['label']).view(-1)
        else:
            return title_enc, toc_enc, intro_enc


    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def label_size(self):
        return self._label_size

    @property
    def text_encoder(self):
        return self._text_encoder

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def vocab(self):
        return self._vocab 


    def create_vocab(self, remove_less_freq_words = True, threshold = 1):
        '''
        Creates vocabulary from the dataset tokens

        Returns:
        List of unique tokens in dataset
        '''
        temp_vocab = []
        for sample in self.data['data']:
            temp_vocab.extend(sample['title'].split())
            temp_vocab.extend(' '.join(sample['toc']).split())
            temp_vocab.extend(sample['intro'].split())
        
        vocab = []
        
        if remove_less_freq_words: 
            
            count_dict = collections.Counter(temp_vocab)
            
            for word in count_dict.keys():
                if count_dict[word] > threshold:
                    vocab.append(word)
                    
        else:
            vocab = sorted(list(set(temp_vocab)))
                
        return vocab


    def split(self, x):
        '''
        Splits the text into tokens 
        '''
        return x.split()


    def collate_fn(self, batch, padding=True):
        """
        Collate function needs to be passed to the pytorch dataloader

        Returns:
        (title,title_lengths): tuple containing padded sequence tensor for title and sequence lengths 
        (toc,toc_lengths): tuple containing padded sequence tensor for table of contents and sequence lengths 
        (intro,intro_lengths): tuple containing padded sequence tensor for introduction and sequence lengths 
        labels: tensor containing labels for the batch
        """
        if self.mode == 'train':
            title, toc, intro, labels = zip(*batch)
            labels = torch.cat(labels)
        else:
            title, toc, intro = zip(*batch)

        if isinstance(intro, collections.Sequence):

            if padding:
                title, title_lengths = stack_and_pad_tensors(title)
                toc, toc_lengths = stack_and_pad_tensors(toc)
                intro, intro_lengths = stack_and_pad_tensors(intro)

            if self.mode == 'train':
                return (title,title_lengths), (toc, toc_lengths), (intro, intro_lengths), labels
            else:
                return (title,title_lengths), (toc, toc_lengths), (intro, intro_lengths)
        else:
            return batch


    @classmethod
    def fromJsonFile(cls, json_file, text_encoder=None, label_encoder=None, vocab = None, mode = 'train'):
        '''
        Read data from json file

        Arguments:
        json_file: string specifying location to json_file
        '''
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        return cls(json_data, text_encoder, label_encoder, vocab, mode)