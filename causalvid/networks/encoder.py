import numpy as np
import torch
import torch.nn as nn
# import random as rd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderQns(nn.Module):
    def __init__(self, word_dim, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, is_gru=False):

        super(EncoderQns, self).__init__()
        self.is_gru= is_gru
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.q_input_ln = nn.LayerNorm((dim_hidden*2 if bidirectional else dim_hidden), elementwise_affine=False)
        self.input_dropout = nn.Dropout(input_dropout_p)
        if is_gru:
            self.rnn_cell = nn.GRU
        else:
            self.rnn_cell = nn.LSTM
            
        self.embedding = nn.Sequential(nn.Linear(word_dim, dim_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))
        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.embedding[0].weight) 


    def forward(self, qns, qns_lengths):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """

        qns_embed = self.embedding(qns)
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths.cpu(), batch_first=True, enforce_sorted=False)
        if self.is_gru:
            packed_output, hidden = self.rnn(packed)
        else:
            packed_output, (hidden, _) = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        output = self.q_input_ln(output) # bs,q_len,hidden_dim
        return output, hidden


class EncoderVid(nn.Module):
    def __init__(self, dim_vid, d_model, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, is_gru=False, vid_encoder_type='rnn'):
        """
        """
        self.vid_encoder_type = vid_encoder_type
        self.is_gru=is_gru
        super(EncoderVid, self).__init__()
        self.dim_vid = dim_vid
        self.d_model = d_model
        self.input_dropout_p = input_dropout_p
        if vid_encoder_type == 'rnn':
            self.rnn_dropout_p = rnn_dropout_p
            self.n_layers = n_layers
            self.bidirectional = bidirectional
            self.v_input_ln = nn.LayerNorm(d_model, elementwise_affine=False)

            self.vid2hid = nn.Sequential(nn.Linear(self.dim_vid, d_model//2),
                                        nn.ReLU(),
                                        nn.Dropout(input_dropout_p))
            if is_gru:
                self.rnn_cell = nn.GRU
            else:
                self.rnn_cell = nn.LSTM

            self.rnn = self.rnn_cell(d_model//2, d_model//2, n_layers, batch_first=True,
                                    bidirectional=True, dropout=self.rnn_dropout_p)

            self._init_weight()
        
        elif vid_encoder_type == 'cnn':
            self.vid_encoder = nn.Sequential(nn.Conv1d(dim_vid, d_model, 3, padding=1),nn.ELU(inplace=True))


    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight) 


    def forward(self, vid_feats, v_len = None):
        """
        vid_feats: (bs, 16, 4096)
        fg_mask: (bs, 16,) bool mask
        """
        if self.vid_encoder_type == 'rnn':
            batch_size, seq_len, dim_vid = vid_feats.size()
            vid_feats_trans = self.vid2hid(vid_feats.view(-1, self.dim_vid))
            vid_feats = vid_feats_trans.view(batch_size, seq_len, -1)

            if v_len is not None:
                vid_feats = pack_padded_sequence(vid_feats, v_len.cpu(), batch_first=True, enforce_sorted=False)

            # self.rnn.flatten_parameters() # for parallel
            if self.is_gru:
                foutput, fhidden = self.rnn(vid_feats)
            else:
                foutput, (fhidden,_) = self.rnn(vid_feats)

            if v_len is not None:
                foutput, _ = pad_packed_sequence(foutput, batch_first=True)

            v_global = torch.cat([fhidden[0], fhidden[1]], dim=-1)
            v_local = self.v_input_ln(foutput) # bs,16,hidden_dim

        elif self.vid_encoder_type == 'cnn':
            v_local = self.vid_encoder(vid_feats.transpose(1, 2)).transpose(1, 2)
            v_global = v_local.mean(1)

        return v_local, v_global