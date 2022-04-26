import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from know_model import DAGKNN
from model import CausePredictor


class UtterEncoder2(nn.Module):
    def __init__(self, model_size, utter_dim, conv_encoder='none', rnn_dropout=None):
        super(UtterEncoder2, self).__init__()
        encoder_path = 'roberta-' + model_size
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        if model_size == 'base':
            token_dim = 768
        else:
            token_dim = 1024
        self.mapping = nn.Linear(token_dim, utter_dim)
        if conv_encoder == 'none':
            self.register_buffer('conv_encoder', None)
        else:
            self.conv_encoder = nn.GRU(input_size=utter_dim,
                                       hidden_size=utter_dim,
                                       bidirectional=False,
                                       num_layers=1,
                                       dropout=rnn_dropout,
                                       batch_first=True)

    def forward(self, conv_utterance, attention_mask, conv_len):
        # conv_utterance: [[conv_len1, max_len1], [conv_len2, max_len2], ..., [conv_lenB, max_lenB]]
        processed_output = []
        for cutt, amsk in zip(conv_utterance, attention_mask):
            output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state
            # [conv_len, token_dim] -> [conv_len, utter_dim]
            pooler_output = torch.max(output_data, dim=1)[0]
            mapped_output = self.mapping(pooler_output)
            processed_output.append(mapped_output)
        # [batch_size, conv_size, utter_dim]
        conv_output = pad_sequence(processed_output, batch_first=True)
        if self.conv_encoder is not None:
            pad_conv = pack_padded_sequence(conv_output, conv_len, batch_first=True, enforce_sorted=False)
            pad_output = self.conv_encoder(pad_conv)[0]
            conv_output = pad_packed_sequence(pad_output, batch_first=True)[0]
        return conv_output


class CskCauseDag2(nn.Module):
    def __init__(self,
                 model_size,
                 utter_dim,
                 conv_encoder,
                 rnn_dropout,
                 num_layers,
                 dropout,
                 pooler_type,
                 add_emotion,
                 emotion_emb,
                 emotion_dim):
        super(CskCauseDag2, self).__init__()
        self.utter_encoder = UtterEncoder2(model_size, utter_dim, conv_encoder, rnn_dropout)
        self.dag = DAGKNN(utter_dim, utter_dim, num_layers, dropout, pooler_type)
        self.classifier = CausePredictor(utter_dim, utter_dim)
        if model_size == 'base':
            csk_dim = 768
        else:
            csk_dim = 1024
        self.csk_mapping = nn.Linear(csk_dim, utter_dim)
        self.add_emotion = add_emotion
        if add_emotion:
            self.emotion_embeddings = nn.Embedding(emotion_emb.shape[0], emotion_emb.shape[1], padding_idx=0, _weight=emotion_emb)
            self.emotion_lin = nn.Linear(emotion_emb.shape[1], emotion_dim)
            self.emotion_mapping = nn.Linear(emotion_dim + utter_dim, utter_dim)
        else:
            self.emotion_embeddings = None
            self.emotion_lin = None
            self.emotion_mapping = None

    def forward(self, input_ids, attention_mask, conv_len, mask, s_mask, o_mask,
                e_mask, emotion_label, knowledge_text, knowledge_mask, know_adj):
        utter_emb = self.utter_encoder(input_ids, attention_mask, conv_len)
        # (num_know, seq_len, bert_dim)
        knowledge_emb = self.utter_encoder.encoder(knowledge_text, attention_mask=knowledge_mask).last_hidden_state
        # (num_know, bert_dim)
        knowledge_emb = torch.max(knowledge_emb, dim=1)[0]
        knowledge = F.relu(self.csk_mapping(knowledge_emb))
        if self.add_emotion:
            emo_emb = self.emotion_lin(self.emotion_embeddings(emotion_label))
            utter_emb = self.emotion_mapping(torch.cat([utter_emb, emo_emb], dim=-1))
        utter_emb = self.dag(utter_emb, knowledge, e_mask, s_mask, o_mask, know_adj)
        logits = self.classifier(utter_emb, mask)
        return logits
