import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausePredictor, mask_logic
from ck_transformer import UtterEncoder2


class CskCauseDag0(nn.Module):
    def __init__(self,
                 model_size,
                 mapping_type,
                 utter_dim,
                 conv_encoder,
                 rnn_dropout,
                 num_layers,
                 dropout,
                 pooler_type,
                 add_emotion,
                 emotion_emb,
                 emotion_dim):
        super(CskCauseDag0, self).__init__()
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


class DAGKNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, pooler_type='all'):
        super(DAGKNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.pooler_type = pooler_type
        for i in range(self.num_layers):
            self.layers.append(GRNKLayer(input_size, hidden_size))
        if self.pooler_type == 'all':
            self.emb_lin = nn.Linear((num_layers+1)*hidden_size, hidden_size)
        else:
            self.emb_lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, features, knowledge, adj, s_mask, o_mask, know_adj):
        H = [features]
        H1 = features
        for i in range(self.num_layers):
            H1 = self.dropout(self.layers[i](H1, knowledge, adj, s_mask, o_mask, know_adj))
            H.append(H1)
        if self.pooler_type == 'all':
            H = torch.cat(H, dim=2)
            utter_emb = self.emb_lin(H)
        else:
            utter_emb = self.emb_lin(H[-1])
        return utter_emb


class GRNKLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRNKLayer, self).__init__()
        self.gru_c = nn.GRUCell(input_size, hidden_size)
        self.gru_p = nn.GRUCell(input_size, hidden_size)
        self.gru_s = nn.GRUCell(input_size, hidden_size)
        self.gat = GraphAttentionK(hidden_size)

    def forward(self, features, knowledge, adj, s_mask, o_mask, know_adj):
        # features: [batch_size, num_utter, utter_dim]
        num_utter = features.size()[1]
        # the first utterance
        # [batch_size, 1, utter_dim]
        C = self.gru_c(features[:, 0, :]).unsqueeze(1)
        M = torch.zeros_like(C).squeeze(1)
        self_know = torch.index_select(knowledge, 0, know_adj[:, 0, 0])
        KS = self.gru_s(self_know, features[:, 0, :]).unsqueeze(1)
        # [batch_size]
        P = self.gru_p(M, features[:, 0, :]).unsqueeze(1)
        H1 = C + P + KS
        for i in range(1, num_utter):
            _, M = self.gat(features[:, i, :], H1, H1, adj[:, i, :i],
                               s_mask[:, i, :i], o_mask[:, i, :i])
            C = self.gru_c(features[:, i, :], M).unsqueeze(1)
            self_know = torch.index_select(knowledge, 0, know_adj[:, i, i])
            KS = self.gru_s(self_know, features[:, i, :]).unsqueeze(1)
            P = self.gru_p(M, features[:, i, :]).unsqueeze(1)
            H_temp = C + P + KS
            # [batch_size, i+1, utter_dim]
            H1 = torch.cat((H1, H_temp), dim=1)
        return H1


class GraphAttentionK(nn.Module):
    def __init__(self, hidden_size):
        super(GraphAttentionK, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, Q, K, V, adj, s_mask, o_mask):
        B = K.size()[0]
        N = K.size()[1]
        Q = Q.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        X = torch.cat((Q, K), dim=2)  # (B, N, 2D)
        alpha = self.linear(X).permute(0, 2, 1)  # (B, 1, N)
        adj = adj.unsqueeze(1)  # (B, 1, N)
        alpha = mask_logic(alpha, adj)  # (B, 1, N)

        attn_weight = F.softmax(alpha, dim=2)  # (B, 1, N)

        V0 = self.Wr0(V)  # (B, N, D)
        V1 = self.Wr1(V)  # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()  # (B, N, 1)
        o_mask = o_mask.unsqueeze(2).float()
        V = V0 * s_mask + V1 * o_mask

        attn_sum = torch.matmul(attn_weight, V).squeeze(1)  # (B, D)

        return attn_weight, attn_sum
