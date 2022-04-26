import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DAGNN, CausePredictor
from encoder import UtterEncoder2


class CauseDag(nn.Module):
    def __init__(self,
                 model_size,
                 mapping_type,
                 utter_dim,
                 conv_encoder,
                 rnn_dropout,
                 num_layers,
                 dropout,
                 pooler_type,
                 emotion_emb,
                 emotion_dim):
        super(CauseDag, self).__init__()
        # self.utter_encoder = UtterEncoder(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout)
        self.utter_encoder = UtterEncoder2(model_size, utter_dim, conv_encoder, rnn_dropout)
        self.dag = DAGNN(utter_dim, utter_dim, num_layers, dropout, pooler_type)
        self.classifier = CausePredictor(utter_dim, utter_dim)

        self.emotion_embeddings = nn.Embedding(emotion_emb.shape[0], emotion_emb.shape[1], padding_idx=0, _weight=emotion_emb)
        self.emotion_lin = nn.Linear(emotion_emb.shape[1], emotion_dim)
        self.emotion_mapping = nn.Linear(emotion_dim + utter_dim, utter_dim)

    def forward(self, input_ids, attention_mask, conv_len, mask, s_mask, o_mask, e_mask, emotion_label):
        utter_emb = self.utter_encoder(input_ids, attention_mask, conv_len)
        emo_emb = self.emotion_lin(self.emotion_embeddings(emotion_label))
        utter_emb = self.emotion_mapping(torch.cat([utter_emb, emo_emb], dim=-1))
        utter_emb = self.dag(utter_emb, e_mask, s_mask, o_mask)
        logits = self.classifier(utter_emb, mask)
        return logits
