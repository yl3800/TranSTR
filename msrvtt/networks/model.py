from builtins import print, tuple
from signal import pause
import torch
import torch.nn as nn
# import random as rd
import torch.nn.functional as F
from itertools import chain
# import difftopk

import os
import sys
sys.path.append('../')
from einops import rearrange, repeat
from networks.util import length_to_mask
from networks.multimodal_transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from networks.position_encoding import PositionEmbeddingSine1D
from transformers import AutoModel, AutoTokenizer
from networks.topk import HardtopK, PerturbedTopK

# from networks.encoder import EncoderVid
# from block import fusions #pytorch >= 1.1.0

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class VideoQAmodel(nn.Module):
    def __init__(self, text_encoder_type="roberta-base", freeze_text_encoder = False, n_query=5,
                        objs=20, frames=16, topK_frame=4, topK_obj=5, hard_eval=False, **kwargs):
        super(VideoQAmodel, self).__init__()
        self.d_model = kwargs['d_model']
        encoder_dropout = kwargs['encoder_dropout']
        self.mc = n_query
        self.hard_eval = hard_eval
        # text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.obj_resize = FeatureResizer(
            input_feat_size=2052,
            output_feat_size=self.d_model, 
            dropout=kwargs['dropout'])

        self.frame_topK, self.obj_topK = topK_obj, topK_frame
        self.frame_sorter = PerturbedTopK(self.frame_topK)
        self.obj_sorter = PerturbedTopK(self.obj_topK)

        # hierarchy 1: obj & frame
        self.obj_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.frame_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.fo_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        
        self.vl_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.ans_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))

        # position embedding
        self.pos_encoder_1d = PositionEmbeddingSine1D()

        # cls head
        self.classifier=nn.Linear(self.d_model, 5000+1) # ans_num+<unk>

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p not in self.text_encoder.parameters():
    #             # if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
                

    def forward(self, frame_feat, obj_feat, qns_word):
        """
        :param vid_frame_feat:[bs, 8, 2, 768]
        :param vid_obj_feat:[bs, 16, 5, 2048]
        :param qns: ('what are three people sitting on?', 'what is a family having?')
        :return:
        """
        # Size
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        # encode q
        q_local, q_mask = self.forward_text(list(qns_word), device)  # [batch, q_len, d_model]
        tgt = ((q_local * (q_mask.float().unsqueeze(-1))).sum(1))/(q_mask.float().sum(-1).unsqueeze(-1)) # mean pooling

        #### encode v
        # frame
        frame_mask = torch.ones(B, F).bool().to(device)
        frame_local, frame_att = self.frame_decoder(frame_feat,
                                    q_local,
                                    memory_key_padding_mask=q_mask,
                                    query_pos = self.pos_encoder_1d(frame_mask , self.d_model),
                                    output_attentions=True
                                    ) # b,16,d
        
        if self.training:
            idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk
        else:
            if self.hard_eval:
                idx_frame = rearrange(HardtopK(frame_att.flatten(1,2), self.frame_topK), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk
            else:
                idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk

        frame_local = (frame_local.transpose(1,2) @ idx_frame).transpose(1,2) # B, Frame_K, d)

        # obj
        obj_feat = (obj_feat.flatten(-2,-1).transpose(1,2) @ idx_frame).transpose(1,2).view(B,self.frame_topK,O,-1)
        obj_local = self.obj_resize(obj_feat)
        obj_local, obj_att = self.obj_decoder(obj_local.flatten(0,1),
                                            q_local.repeat_interleave(self.frame_topK, dim=0), 
                                            memory_key_padding_mask=q_mask.repeat_interleave(self.frame_topK, dim=0),
                                            output_attentions=True
                                            )  # b*16,5,d        #.view(B, F, O, -1) # b,16,5,d

        if self.training:
            idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
        else:
            if self.hard_eval:
                idx_obj = rearrange(HardtopK(obj_att.flatten(1,2), self.obj_topK), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
            else:
                idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
        obj_local = (obj_local.transpose(1,2) @ idx_obj).transpose(1,2).view(B, self.frame_topK, self.obj_topK, -1)


        ### hierarchy grouping
        # interframe
        # frame_obj = self.fo_decoder(frame_local.flatten(0,1).unsqueeze(1), obj_local.flatten(0,1)) # b,16,d
        # cross frame
        frame_obj = self.fo_decoder(frame_local, obj_local.flatten(1,2)) # b,16,d

        ### overall fusion
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        frame_obj =frame_obj.view(B, self.frame_topK, -1)
        frame_qns_mask = torch.cat((frame_mask, q_mask),dim=1).bool()
        mem = self.vl_encoder(torch.cat((frame_obj, q_local), dim=1), \
                            src_key_padding_mask=frame_qns_mask, \
                            pos = self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model)
                            ) # b,16,d
        
        # predict
        out = self.ans_decoder(tgt.unsqueeze(1), mem, memory_key_padding_mask=frame_qns_mask.bool())
        out = self.classifier(out).squeeze(1) # 这里squeeze是由于classifier会出来最后一维是1
        return out
        

    def forward_text(self, text_queries, device, has_ans=False):
        """
        text_queries : list of question str 
        out: text_embedding: bs, len, dim
            mask: bs, len (bool) [1,1,1,1,0,0]
        """
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        # tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='max_length', 
        #                                                     max_length=self.qa_max_len if has_ans else self.q_max_len, 
        #                                                     return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state

        return encoded_text, tokenized_queries.attention_mask.bool()
    


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="train parameter")
    # general
    parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=256)
    parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
    parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=25)
    parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
    parser.add_argument("-es", action="store_true", help="early_stopping")
    parser.add_argument("-dropout", "-drop", type=float, help="dropout rate", default=0.2)  
    parser.add_argument("-patience", "-pa", type=int, help="patience of ReduceonPleatu", default=5)  
    parser.add_argument("-encoder_dropout", "-ep", type=float, help="dropout rate", default=0.3)   

    # dataset
    parser.add_argument('-dataset', default='msrvtt-qa',choices=['msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument("-ans_num", type=int, help="ans vocab num", default=5000)
    parser.add_argument("-n_query", type=int, help="multi-choice", default=5)  

    # model
    parser.add_argument("-is_gru", action="store_true", help="gru or lstm in Qns Encoder")
    parser.add_argument("-d_model", "-md",  type=int, help="hidden dim of vq encoder", default=768) 
    parser.add_argument("-word_dim", "-wd", type=int, help="word dim ", default=768)   
    parser.add_argument("-vid_dim", "-vd", type=int, help="vis dim", default=2048) 
    parser.add_argument('-vid_encoder_type', "-ve", default='cnn',choices=['rnn', 'cnn'], type=str)
    parser.add_argument("-hard_eval", "-hd", action="store_true", help="hard selection during inference")
    parser.add_argument("-topK_frame", "-fk", type=int, help="word dim ", default=8)   
    parser.add_argument("-topK_obj", "-ok", type=int, help="word dim ", default=5) 

    # transformer
    parser.add_argument("-trans_hid", type=int, help="hidden dim of ffn in transfomer", default=2048) 
    parser.add_argument("-num_encoder_layers", "-el", type=int, help="number of encoder layers in transformer", default=2)
    parser.add_argument("-num_decoder_layers", "-dl", type=int, help="number of decoder layers in transformer", default=2)
    parser.add_argument("-nheads", type=int, help="num of attention head", default=8) 
    parser.add_argument("-normalize_before", action="store_true", help="pre or post normalize")
    parser.add_argument("-activation", default='relu', choices=['relu','gelu','glu'], type=str)
    parser.add_argument("-return_intermediate", "-ri", action="store_true", help="return intermediate of decoder")
    
    # lan model
    parser.add_argument("-freeze_text_encoder", action="store_true", help="freeze text encoder")
    parser.add_argument("-text_encoder_type", "-t", default="roberta-base", \
                        choices=["roberta-base","distilroberta-base","bert-base-uncased",\
                            "distilbert-base-uncased","microsoft/deberta-base"], type=str)
    parser.add_argument('-text_pool_mode',"-pool", default=0, choices=[0,1,2],help="0last hidden, 1mean, 2max", type=int)

    args = parser.parse_args()
    config = {**vars(args)}
    # print(config)
    # videos=torch.rand(2,8,4096)
    vid_obj_feat = torch.rand(2,16, 10, 2048+4)
    vid_frame_feat = torch.rand(2, 16, 768)
    qns = ('what are three people sitting on?', 'what is a family having?')


    model=VideoQAmodel(**config)
    # model.eval()
    model.to('cuda')
    vid_frame_feat = vid_frame_feat.to('cuda')
    vid_obj_feat = vid_obj_feat.to('cuda')
    out = model(vid_frame_feat, vid_obj_feat, qns)
    print(out.shape)

    # parameters
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameter: %.2fM" % (total/1e6))