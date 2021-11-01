import torch
import torch.nn as nn
from torchcrf import CRF
from itertools import repeat


class Spatial_Dropout(nn.Module):
    def __init__(self,drop_prob):

        super(Spatial_Dropout,self).__init__()
        self.drop_prob = drop_prob

    def forward(self,inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self,input):
        return input.new().resize_(input.size(0),*repeat(1, input.dim() - 2),input.size(2))


def create_emb_layer(weights_matrix, is_trainable=True):
	num_embeddings, emb_dim = weights_matrix.shape
	emb_layer = nn.Embedding(num_embeddings, emb_dim)
	emb_layer.weight = nn.Parameter(torch.from_numpy(weights_matrix))
	if not is_trainable:
		emb_layer.weight.requires_grad = False
	return emb_layer, num_embeddings, emb_dim


class BiLSTM_CRF(nn.Module):

    def __init__(self, weights_matrix, rnn_type, num_layers, rnn_dim, dropout, num_labels, is_trainable=True):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding, _, emb_dim = create_emb_layer(weights_matrix, is_trainable)
        self.embedding.float()
        
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=emb_dim, 
                                hidden_size=rnn_dim, 
                                num_layers=num_layers, 
                                bidirectional=True, 
                                batch_first=False,
                                dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(input_size=emb_dim, 
                                hidden_size=rnn_dim, 
                                num_layers=num_layers, 
                                bidirectional=True, 
                                batch_first=False,
                                dropout=dropout if num_layers > 1 else 0.0)
        out_dim = rnn_dim * 2
        self.spatial_dropout = Spatial_Dropout(0.2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(emb_dim, eps=1e-05)
        self.layer_norm2 = nn.LayerNorm(out_dim, eps=1e-05)
        self.hidden2tag = nn.Linear(out_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    
    
    def forward(self, input_ids, input_mask=None, tags=None):
        emissions = self.tag_outputs(input_ids, input_mask)
        prediction = self.crf.decode(emissions, input_mask.byte())
        if tags is not None:
            loss = -1*self.crf(emissions, tags, mask=input_mask.byte())
            return loss, prediction
        else:
            return prediction

    
    def tag_outputs(self, input_ids, input_mask=None):

        embeddings = self.embedding(input_ids)
        embeddings = self.spatial_dropout(embeddings)
        embeddings = self.layer_norm1(embeddings)
        
        input_lengths = torch.sum(input_mask==1, dim=-1).detach().cpu().numpy()
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, 
                                                          lengths=input_lengths, 
                                                          enforce_sorted=False, 
                                                          batch_first=True)
        
        sequence_output, _ = self.rnn(embeddings)
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(sequence_output)
        sequence_output = sequence_output.permute(1, 0, 2)
        
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.layer_norm2(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions

