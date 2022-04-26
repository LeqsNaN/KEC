import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel
from torch.nn.utils.rnn import pad_sequence
from model import mask_logic, CausePredictor


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, nhead, dropout=0.1, attn_mask=False):
        super(MultiHeadAttention, self).__init__()
        self.attn_mask = attn_mask
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)

        self.o_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, q, k, v, mask):
        src_len = q.size(0)
        tgt_len = k.size(0)

        assert src_len == tgt_len, "length of query does not equal length of key"

        scaling = float(self.head_dim) ** -0.5

        query = F.linear(q, self.q_proj_weight)
        key = F.linear(k, self.k_proj_weight)
        value = F.linear(v, self.v_proj_weight)

        # (n_head, s_len, h_dim)
        query = query.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)

        # q*k
        attn_weight = torch.matmul(query, key.transpose(1, 2))
        attn_weight = attn_weight * scaling

        if mask is not None:
            # [1, s_len, s_len]
            mask = mask.unsqueeze(0).expand(self.nhead, src_len, tgt_len)
            attn_weight = torch.masked_fill(attn_weight, mask, -1e30)
        # (n_head, src_len, tgt_len)
        attn_score = F.softmax(attn_weight, dim=-1)
        if self.attn_mask:
            attmask = mask.eq(False).to(torch.float).unsqueeze(0)
            attn_score = attn_score * attmask
        attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_score, value)
        # (n_head, src_len, h_dim) -> (src_len, n_head, h_dim) -> (src_len, emb_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(src_len, -1)
        output = F.linear(attn_output, self.o_proj.weight)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, attn_mask=False):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, nhead, dropout, attn_mask)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src2 = self.attention(src, src, src, src_mask)
        ss = src + self.dropout1(src2)
        ss = self.norm1(ss)
        ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        return ss


class UtterEncoder3(nn.Module):
    def __init__(self, model_size, utter_dim, num_layers, nhead, ff_dim, att_dropout):
        super(UtterEncoder3, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-'+model_size)
        if model_size == 'base':
            bert_dim = 768
        else:
            bert_dim = 1024
        self.trm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerLayer(utter_dim, nhead, ff_dim, att_dropout)
            self.trm_layers.append(layer)
        self.num_layers = num_layers
        self.attn = nn.Linear(utter_dim, 1)
        self.utter_mapping = nn.Linear(bert_dim, utter_dim)

    def forward(self, conv_utterance, attention_mask):
        processed_output = []
        doc_output = []
        for cutt, amsk in zip(conv_utterance, attention_mask):
            output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state
            pooler_output = torch.max(output_data, dim=1)[0]
            # [num_utter, dim]
            mapped_output = self.utter_mapping(pooler_output)
            src_mask = torch.ones((mapped_output.shape[0], mapped_output.shape[0])).tril(0).eq(0)
            src_mask = src_mask.to(mapped_output.device)
            for i in range(self.num_layers):
                mapped_output = self.trm_layers[i](mapped_output, src_mask)
            processed_output.append(mapped_output)
            attention_weight = torch.softmax(self.attn(mapped_output), dim=0)
            doc_output.append(torch.matmul(attention_weight.transpose(0, 1), mapped_output))
        return processed_output, doc_output


class EdgeGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(EdgeGCN, self).__init__()
        # num_bases=2=num_relations, no base is applied according to dgl.
        self.in_features = in_features
        self.sequence_weight = nn.Linear(in_features, out_features, False)
        self.knowledge_weight = nn.Linear(in_features, out_features, False)
        self.self_loop_weight = nn.Linear(in_features, out_features, False)

    def norm(self, adj):
        # directed graph
        deg_inv = (adj.sum(-1) + 1e-10).pow(-1)
        adj = adj * deg_inv.unsqueeze(2)
        return adj

    def forward(self, utt_emb, edge_rep, binary_knowledge_adj, sequence_adj):
        # [batch_size, seq_len, dim]
        zi = self.knowledge_weight(utt_emb)
        # [batch_size, 1, seq_len, dim] + [batch_size, seq_len, seq_len, dim]
        query = self.knowledge_weight(utt_emb.unsqueeze(1) + edge_rep)
        # [batch_size, seq_len, 1, dim]
        key = zi.unsqueeze(2)
        value = zi
        knowledge_adj = (query * key).sum(3) / math.sqrt(self.in_features)
        knowledge_adj = mask_logic(knowledge_adj, binary_knowledge_adj)
        # [batch_size, seq_len, seq_len]
        knowledge_adj = torch.softmax(knowledge_adj, dim=1) * binary_knowledge_adj
        zi = torch.matmul(knowledge_adj.transpose(1, 2), value)
        si = self.sequence_weight(utt_emb)
        sequence_adj = self.norm(sequence_adj)
        si = torch.matmul(sequence_adj, si)
        li = self.sequence_weight(utt_emb)
        x = zi + si + li
        return F.selu(x)


class Kag(nn.Module):
    def __init__(self, model_size, utter_dim, num_layers, nhead, trm_num_layers,
                 add_emotion=True, emotion_emb=None, emotion_dim=200):
        super(Kag, self).__init__()
        self.utter_dim = utter_dim
        self.utter_encoder = UtterEncoder3(model_size, utter_dim, trm_num_layers, nhead, utter_dim, 0.1)
        self.know_mapping = nn.Linear(768, utter_dim)
        self.layers = nn.ModuleList()
        self.add_emotion = add_emotion
        self.num_layers = num_layers
        if add_emotion:
            self.emotion_embedding = nn.Embedding(emotion_emb.shape[0], emotion_emb.shape[1], _weight=emotion_emb)
            self.emotion_lin = nn.Linear(emotion_emb.shape[1], emotion_dim)
            self.emotion_mapping = nn.Linear(emotion_dim+utter_dim, utter_dim)
        else:
            self.emotion_embedding = None
            self.emotion_lin = None
            self.emotion_mapping = None
        for i in range(num_layers):
            layer = EdgeGCN(utter_dim, utter_dim)
            self.layers.append(layer)
        self.classifier = CausePredictor(utter_dim, utter_dim)

    def forward(self, utterance_text, attention_mask, conv_len, knowledge_text,
                batch_know_len, know_mask, know_len, know_adj, seq_adj, mask, emotion_label):
        conv_output, doc_output = self.utter_encoder(utterance_text, attention_mask)
        utter_emb = pad_sequence(conv_output, batch_first=True)
        if self.add_emotion:
            emo_emb = self.emotion_lin(self.emotion_embedding(emotion_label))
            utter_emb = self.emotion_mapping(torch.cat([utter_emb, emo_emb], dim=-1))
        batch_size, seq_len, utter_dim = utter_emb.shape
        know_emb = self.utter_encoder.encoder(knowledge_text, know_mask).last_hidden_state
        know_emb = self.know_mapping(torch.max(know_emb, dim=1)[0])
        know_emb = torch.split(know_emb, batch_know_len, 0)
        know_attn = [torch.zeros(1, self.utter_dim).to(utter_emb.device)]
        for kemb, demb, klen in zip(know_emb, doc_output, know_len):
            # kemb: [num_know, dim]; demb: [1, dim]
            attn = (demb * kemb).sum(1).split(klen)
            kemb_s = kemb.split(klen)
            for a, k in zip(attn, kemb_s):
                # [1, num_know]
                a = torch.softmax(a.unsqueeze(0), dim=1)
                k = torch.matmul(a, k)
                know_attn.append(k)
        # [num_know, dim]
        know_attn = torch.cat(know_attn, dim=0)
        # [batch_size, seq_len, seq_len]
        # [batch_size*seq_len*seq_len, dim]
        edge_rep = torch.index_select(know_attn, 0, know_adj.flatten())
        edge_rep = edge_rep.contiguous().view(batch_size, seq_len, seq_len, utter_dim)
        binary_know_adj = torch.clamp(know_adj.clone().float(), max=1)

        gcn_output = utter_emb
        for i in range(self.num_layers):
            gcn_output = self.layers[i](gcn_output, edge_rep, binary_know_adj, seq_adj)
        logits = self.classifier(gcn_output, mask)
        return logits
