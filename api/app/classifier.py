import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    '''
    Creates a classifier object consisting of Embeddings, LSTM and Dense Layers
    Arguments:
    emb_size: a tuple to initialize embedding layer : vocab_size X embedding_dimension
    n_lstm_layers: number of LSTM layers
    hidden_size: hidden size for LSTM
    num_classes: total number of end classes
    bidirectional: number of directions of the LSTM layer, Default : False
    dense_layer_sizes: list containing sizes(number of neurons) of the dense layers. Also used to determine the number of dense layers in the architecture : [512, 1024, 2048]
    dropout_prob: dropout (same dropout is used with the LSTMs and rest of the architecture), Default: 0.25
    '''
    def __init__(self, emb_size, n_lstm_layers, hidden_size, num_classes, bidirectional = False, dense_layer_sizes = [64, 32], dropout_prob = 0.25, ):

        super(LSTMClassifier, self).__init__()

        self.emb_size = emb_size
        self.n_lstm_layers = n_lstm_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.dense_layer_sizes = dense_layer_sizes
        self.num_classes = num_classes

        # Defining hidden layers
        ## Embeddings
        self.embeddings = nn.Embedding(self.emb_size[0] , self.emb_size[1], padding_idx=0)


        # Encoders for Inputs
        # For Title
        self.title_enc = NonSequentialEncoder(self.emb_size[1], self.hidden_size)

        ## For Table of Contents
        self.toc_enc = NonSequentialEncoder(self.emb_size[1], self.hidden_size)

        ## For Introduction
        self.seq_enc = SequentialEncoder(self.emb_size[1], self.hidden_size, self.n_lstm_layers, self.dropout_prob, self.bidirectional)

        ## Dense Layer to combine the embeddings
        self.combiner = nn.Linear(3, 1, bias = True)


        ## FC layers
        self.dense_sizes = []
        for i in range(len(dense_layer_sizes)):
            self.dense_sizes.append({'input_' + str(i) : self.hidden_size if i == 0 else dense_layer_sizes[i-1],
                                    'output_' + str(i) : dense_layer_sizes[i]})


        dense_layers = []
        for i in range(len(self.dense_sizes)):
            layer = nn.Linear(self.dense_sizes[i]['input_' + str(i)], self.dense_sizes[i]['output_' + str(i)],bias = True)
            exec('self.dense_' + "%s" % (i) + "=%s" % (layer))
            exec('dense_layers.append(self.dense_%s)' % (i))
        self.dense_layers = dense_layers


        ## Last FC layer for softmax
        self.last_dense_layer = nn.Linear(self.dense_sizes[len(self.dense_sizes)-1]['output_' + str(len(self.dense_sizes)-1)], self.num_classes, bias = True)

        self.dropout = nn.Dropout(p = self.dropout_prob)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, title_attrs, toc_attrs, intro_attrs):

        # Obtain embeddings for all inputs
        title_embs = self.get_embs_non_sequential_input(title_attrs[0], title_attrs[1])
        title_embs = self.title_enc(title_embs)
        
        toc_embs = self.get_embs_non_sequential_input(toc_attrs[0], toc_attrs[1])
        toc_embs = self.toc_enc(toc_embs)

        intro_embs = self.embeddings(intro_attrs[0])
        intro_embs = self.seq_enc(intro_embs, intro_attrs[1])

        
        # Concatenate embeddings from all input types
        output = torch.cat([title_embs, toc_embs, intro_embs], dim = 2)
        output = self.combiner(output).squeeze(2)
        output = self.dropout(output)


        # Forward pass through Dense layers
        for layer in self.dense_layers:
            output = layer(output)
            output = self.relu(output)
            output = self.dropout(output)  


        # Last dense layer for softmax scores
        output = self.last_dense_layer(output)

        return output


    def get_embs_non_sequential_input(self, inp, inp_lengths):

        embs = self.embeddings(inp)

        sorted_lengths, sorted_index = inp_lengths.sort(dim = 0, descending = True)
        embs = embs[sorted_index]

        embs = nn.utils.rnn.pack_padded_sequence(embs, sorted_lengths, batch_first = True)
        embs, _ = nn.utils.rnn.pad_packed_sequence(embs, batch_first = True)

        return embs.sum(dim = 1)


class NonSequentialEncoder(nn.Module):
    '''
    Encoder for non-sequential inputs (title and table of contents of wiki page)

    Arguments:
    emb_dim: dimensions of the embedding
    out_dim: output dimensions of the encoder
    '''
    def __init__(self, emb_dim, out_dim):

        super(NonSequentialEncoder, self).__init__()

        self.dense = nn.Linear(emb_dim, out_dim, bias = True)
        self.relu = nn.ReLU()


    def forward(self, inp):
        
        output = self.dense(inp)
        output = self.relu(output)

        return output.unsqueeze(2)

class SequentialEncoder(nn.Module):
    '''
    Encoder for sequential inputs (Introduction section of wiki page)

    Arguments:
    emb_dim: dimensions of the embedding
    hidden_size: size hidden units of LSTM layer
    n_lstm_layers: number of LSTM layers stacked together
    dropout_prob: dropout, Default: 0.25
    bidirectional: boolean If True, uses bidirectional LSTM
    '''
    def __init__(self, emb_dim, hidden_size, n_lstm_layers = 1, dropout_prob=0.25, bidirectional = False):

        super(SequentialEncoder, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, self.n_lstm_layers, dropout = self.dropout_prob, batch_first = True, bidirectional = self.bidirectional)


    def forward(self, x, seq_lengths):
        
        # Sort the sequences in descending order of lengths. This is required for pack_padded_sequence. 
        # Not needed while training since we would be using BucketBatchSampler with the DataLoader object which passes sorted input dor the forward propagation
        if  not self.training:
            sorted_lens, sorted_idx = seq_lengths.sort(dim = 0, descending = True)
            x = x[sorted_idx]
        else:
            sorted_lens = seq_lengths


        # Pack Padded Sequence to mask the padded indices
        output = nn.utils.rnn.pack_padded_sequence(x, sorted_lens, batch_first = True)


        # Forward pass through the LSTM layer
        output, (hn, cn) = self.lstm(output)


        # Obtain padded sequence from the packed sequence
        output,_ = nn.utils.rnn.pad_packed_sequence(output, batch_first = True)


        # Restore batch sequence using sorted_idx defined above. This part is required only during inference
        if not self.training:
            _, idx_to_restore = sorted_idx.sort(dim = 0, descending = False)
            output = output.gather(0, idx_to_restore.view(-1, 1, 1).expand_as(output))


        # Aggregate the hidden representations of forward and backward directions of LSTM. 
        if self.bidirectional:
            n_directions = 2
            output = output.contiguous().view(-1, sorted_lens[0].item(), n_directions, self.hidden_size)
            output = torch.sum(output, dim = 2)


        # Extract output corresponding to last timestep
        time_dimension = 1
        idx = (seq_lengths-1).view(-1,1).expand(len(seq_lengths), output.size(2)).unsqueeze(time_dimension)
        idx = idx.to(output.device)
        output = output.gather(time_dimension, idx).squeeze(time_dimension)
        
        return output.unsqueeze(2) 