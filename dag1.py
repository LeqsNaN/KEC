import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DAGNN, CausePredictor
from encoder import UtterEncoder2


class CauseDagNoEmotion(nn.Module):
    def __init__(self,
                 model_size,
                 mapping_type,
                 utter_dim,
                 conv_encoder,
                 rnn_dropout,
                 num_layers,
                 dropout,
                 pooler_type):
        super(CauseDagNoEmotion, self).__init__()
        # self.utter_encoder = UtterEncoder(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout)
        self.utter_encoder = UtterEncoder2(model_size, utter_dim, conv_encoder, rnn_dropout)
        self.dag = DAGNN(utter_dim, utter_dim, num_layers, dropout, pooler_type)
        self.classifier = CausePredictor(utter_dim, utter_dim)

    def forward(self, input_ids, attention_mask, conv_len, mask, s_mask, o_mask, e_mask, emotion_label):
        utter_emb = self.utter_encoder(input_ids, attention_mask, conv_len)
        utter_emb = self.dag(utter_emb, e_mask, s_mask, o_mask)
        logits = self.classifier(utter_emb, mask)
        return logits
