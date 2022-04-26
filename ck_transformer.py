import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import UtterEncoder2
from model import mask_logic, CausePredictor


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        linear_out = self.linear2(F.relu(self.linear1(x)))
        output = self.norm(self.dropout(linear_out) + x)
        return output


class PositionEncoding(nn.Module):
    def __init__(self, input_dim, max_len=200):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (bsz, slen, dim)
        seq_len = x.size(1)
        pe_clip = self.pe[:seq_len]
        pemb = x + pe_clip.unsqueeze(0)
        return pemb


class RelativePositionEncoding(nn.Module):
    def __init__(self, input_dim, max_len=10):
        super(RelativePositionEncoding, self).__init__()
        self.max_len = max_len
        self.pe_k = nn.Embedding(max_len+1, input_dim, padding_idx=0)
        self.pe_v = nn.Embedding(max_len+1, input_dim, padding_idx=0)

    def forward(self, position_mask):
        position_mask = torch.clamp(position_mask, min=0, max=self.max_len).long()
        # (slen, slen, hdim)
        pemb_k = self.pe_k(position_mask)
        pemb_v = self.pe_v(position_mask)
        return pemb_k, pemb_v


class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, emb_dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight_s = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight_o = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.o_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.know_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.q_proj_weight)
        torch.nn.init.xavier_uniform_(self.k_proj_weight)
        torch.nn.init.xavier_uniform_(self.v_proj_weight_s)
        torch.nn.init.xavier_uniform_(self.v_proj_weight_o)
        torch.nn.init.xavier_uniform_(self.o_proj_weight)
        torch.nn.init.xavier_uniform_(self.know_proj_weight)

    def forward(self, x, adj, s_mask, o_mask, know, rel_pos_k=None, rel_pos_v=None, require_weights=False):
        # knowledge: (knum, nh*hdim), know_adj: (bsz, slen, slen)
        # adj, s_mask, o_mask: (bsz, slen, slen)
        # input size: (slen, bsz, nh*hdim)
        slen = x.size(0)
        bsz = x.size(1)

        know = F.linear(know, self.know_proj_weight).view(bsz, slen*slen, self.nhead, self.head_dim)
        know = know.transpose(0, 1).contiguous().view(slen*slen, bsz*self.nhead, self.head_dim)
        know = know.transpose(0, 1).contiguous().view(bsz*self.nhead, slen, slen, self.head_dim)

        scaling = float(self.head_dim) ** -0.5
        query = F.linear(x, self.q_proj_weight)
        key = F.linear(x, self.k_proj_weight)
        value_s = F.linear(x, self.v_proj_weight_s)
        value_o = F.linear(x, self.v_proj_weight_o)

        # (slen, bsz, nh*hdim) -> (slen, bsz*nh, hdim) -> (bsz*nh, slen, slen, hdim)
        query = query.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(2)
        key = key.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)
        # (bsz*nh, 1, slen, hdim)
        value_s = value_s.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)
        value_o = value_o.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)

        # (bsz*nh, slen, slen)
        if rel_pos_k is None:
            attention_weight = query * (key + know)
        else:
            attention_weight = query * (key + know + rel_pos_k)
        attention_weight = attention_weight.sum(3) * scaling
        attention_weight = mask_logic(attention_weight, adj)
        attention_weight = F.softmax(attention_weight, dim=2)
        attention_weight = F.dropout(attention_weight, p=self.dropout, training=self.training)

        value = value_s * s_mask + value_o * o_mask

        # (bsz*nh, slen, slen, hdim) -> (bsz*nh, slen, hdim)
        if rel_pos_v is None:
            attn_sum = (value * attention_weight.unsqueeze(3)).sum(2)
        else:
            attn_sum = ((value + rel_pos_v) * attention_weight.unsqueeze(3)).sum(2)
        attn_sum = attn_sum.transpose(0, 1).contiguous().view(slen, bsz, -1)
        output = F.linear(attn_sum, self.o_proj_weight)

        if require_weights:
            return output, attention_weight
        else:
            return output


class TransformerLayer(nn.Module):
    def __init__(self, emb_dim, nhead, ff_dim, att_dropout, dropout):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(nhead, emb_dim, att_dropout)

        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.ff_net = MLP(emb_dim, ff_dim, dropout)

    def forward(self, x, adj, s_mask, o_mask, know, rel_pos_k=None, rel_pos_v=None, requires_weights=False):
        if requires_weights:
            x2, weight = self.attention(x, adj, s_mask, o_mask, know, rel_pos_k, rel_pos_v, requires_weights)
        else:
            x2 = self.attention(x, adj, s_mask, o_mask, know, rel_pos_k, rel_pos_v, requires_weights)
            weight = None

        ss = x + self.dropout(x2)
        ss = self.norm(ss)
        ff_out = self.ff_net(ss)

        return ff_out, weight


class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_layers,
                 nhead,
                 emb_dim,
                 ff_dim,
                 att_dropout,
                 dropout,
                 max_len,
                 pe_type='rel'):
        super(TransformerEncoder, self).__init__()
        self.pe_type = pe_type
        self.num_layers = num_layers
        self.nhead = nhead
        self.hdim = emb_dim // nhead
        if pe_type == 'abs':
            self.pe = PositionEncoding(emb_dim, max_len)
        else:
            self.pe = RelativePositionEncoding(emb_dim//nhead, max_len)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerLayer(emb_dim, nhead, ff_dim, att_dropout, dropout)
            self.layers.append(layer)

    def forward(self, x, adj, s_mask, o_mask, knowledge, know_adj, pos_mask=None, requires_weight=False):
        weights_list = []
        bsz = x.shape[0]
        slen = x.shape[1]
        if self.pe_type == 'abs':
            x = self.pe(x)
            rel_emb_k = None
            rel_emb_v = None
        else:
            rel_emb_k, rel_emb_v = self.pe(pos_mask)
            rel_emb_k = rel_emb_k.unsqueeze(0).expand(bsz*self.nhead, slen, slen, self.hdim)
            rel_emb_v = rel_emb_v.unsqueeze(0).expand(bsz*self.nhead, slen, slen, self.hdim)
        s_mask = s_mask.unsqueeze(1).expand(bsz, self.nhead, slen, slen)
        s_mask = s_mask.contiguous().view(bsz*self.nhead, slen, slen, 1)
        o_mask = o_mask.unsqueeze(1).expand(bsz, self.nhead, slen, slen)
        o_mask = o_mask.contiguous().view(bsz*self.nhead, slen, slen, 1)

        adj = adj.unsqueeze(1).expand(bsz, self.nhead, slen, slen)
        adj = adj.contiguous().view(bsz*self.nhead, slen, slen)

        know = torch.index_select(knowledge, 0, torch.flatten(know_adj)).contiguous().view(bsz, slen * slen, -1)

        x1 = x.transpose(0, 1)
        for i in range(self.num_layers):
            x1, weights = self.layers[i](x1, adj, s_mask, o_mask, know, rel_emb_k, rel_emb_v, requires_weight)
            if requires_weight:
                weights_list.append(weights)
        x1 = x1.transpose(0, 1)
        if requires_weight:
            return x1, weights_list
        else:
            return x1


class CskTransformer(nn.Module):
    def __init__(self, model_size, utter_dim, conv_encoder, rnn_dropout, num_layers, nhead, ff_dim, att_dropout,
                 trm_dropout, max_len, pe_type, add_emotion=True, emotion_emb=None, emotion_dim=0):
        super(CskTransformer, self).__init__()
        self.utter_encoder = UtterEncoder2(model_size, utter_dim, conv_encoder, rnn_dropout)
        self.transformer_encoder = TransformerEncoder(num_layers, nhead, utter_dim, ff_dim,
                                                      att_dropout, trm_dropout, max_len, pe_type)
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

    def forward(self, conv_utterance, attention_mask, conv_len, adj, s_mask, o_mask, c_mask,
                emotion_label, knowledge_text, knowledge_mask, know_adj, requires_weight=False):
        utter_emb = self.utter_encoder(conv_utterance, attention_mask, conv_len)
        # (num_know, seq_len, bert_dim)
        knowledge_emb = self.utter_encoder.encoder(knowledge_text, attention_mask=knowledge_mask).last_hidden_state
        # (num_know, bert_dim)
        knowledge_emb = torch.max(knowledge_emb, dim=1)[0]
        knowledge = F.relu(self.csk_mapping(knowledge_emb))
        if self.add_emotion:
            emo_emb = self.emotion_lin(self.emotion_embeddings(emotion_label))
            utter_emb = self.emotion_mapping(torch.cat([utter_emb, emo_emb], dim=-1))
        slen = utter_emb.shape[1]
        src_pos = torch.arange(slen).unsqueeze(0)
        tgt_pos = torch.arange(slen).unsqueeze(1)
        # (slen, slen)
        pos_mask = (tgt_pos - src_pos) + 1
        pos_mask = pos_mask.to(utter_emb.device)
        if requires_weight:
            output, weights = self.transformer_encoder(utter_emb, adj, s_mask, o_mask, knowledge, know_adj, pos_mask, requires_weight)
        else:
            output = self.transformer_encoder(utter_emb, adj, s_mask, o_mask, knowledge, know_adj, pos_mask, requires_weight)
            weights = None
        logits = self.classifier(output, c_mask)
        if requires_weight:
            return logits, weights
        else:
            return logits


class TransformerConv(nn.Module):
    def __init__(self, nhead, emb_dim, dropout, set_beta=True):
        super(TransformerConv, self).__init__()
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.set_beta = set_beta
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.know_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        if set_beta:
            self.skip_lin = nn.Linear(emb_dim, emb_dim, bias=False)
            self.beta = nn.Linear(3*emb_dim, 1, bias=False)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.q_proj_weight)
        torch.nn.init.xavier_uniform_(self.k_proj_weight)
        torch.nn.init.xavier_uniform_(self.v_proj_weight)
        torch.nn.init.xavier_uniform_(self.know_proj_weight)
        if self.set_beta:
            torch.nn.init.xavier_uniform_(self.skip_lin.weight)
            torch.nn.init.xavier_uniform_(self.beta.weight)

    def forward(self, x, mask=None, know=None, require_weight=False):
        # knowledge: (knum, nh*hdim), know_adj: (bsz, slen, slen)
        # adj, s_mask, o_mask: (bsz, slen, slen)
        # input size: (slen, bsz, nh*hdim)
        slen = x.size(0)
        bsz = x.size(1)

        know = F.linear(know, self.know_proj_weight).view(bsz, slen * slen, self.nhead, self.head_dim)
        know = know.transpose(0, 1).contiguous().view(slen * slen, bsz * self.nhead, self.head_dim)
        know = know.transpose(0, 1).contiguous().view(bsz * self.nhead, slen, slen, self.head_dim)

        scaling = float(self.head_dim) ** -0.5
        query = F.linear(x, self.q_proj_weight)
        key = F.linear(x, self.k_proj_weight)
        value = F.linear(x, self.v_proj_weight)

        # (slen, bsz, nh*hdim) -> (slen, bsz*nh, hdim) -> (bsz*nh, slen, slen, hdim)
        query = query.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(2)
        key = key.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)
        # (bsz*nh, 1, slen, hdim)
        value = value.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)

        # (bsz*nh, slen, slen)
        attention_weight = query * (key + know)

        attention_weight = attention_weight.sum(3) * scaling
        attention_weight = mask_logic(attention_weight, mask)
        attention_weight = F.softmax(attention_weight, dim=2)
        attention_weight = F.dropout(attention_weight, p=self.dropout, training=self.training)

        # (bsz*nh, slen, slen, hdim) -> (bsz*nh, slen, hdim)
        attn_sum = ((value + know) * attention_weight.unsqueeze(3)).sum(2)
        # (slen, bsz, nh*hdim)
        attn_sum = attn_sum.transpose(0, 1).contiguous().view(slen, bsz, -1)

        if self.set_beta:
            r = self.skip_lin(x)
            # (slen, bsz, 3*nh*hdim) -> (slen, bsz, 1)
            beta_gate = self.beta(torch.cat([attn_sum, r, attn_sum - r], dim=-1))
            beta_gate = F.sigmoid(beta_gate)
            output = beta_gate * r + (1 - beta_gate) * attn_sum
        else:
            output = attn_sum + x

        if require_weight:
            return output, attention_weight
        else:
            return output


class GraphTransformerLayer(nn.Module):
    def __init__(self, emb_dim, nhead, ff_dim, dropout, set_beta):
        super(GraphTransformerLayer, self).__init__()
        self.attention = TransformerConv(nhead, emb_dim, dropout, set_beta)
        self.ff_net = MLP(emb_dim, ff_dim, dropout)

    def forward(self, x, adj, know, requires_weights=False):
        if requires_weights:
            x2, weight = self.attention(x, adj, know, requires_weights)
        else:
            x2 = self.attention(x, adj, know, requires_weights)
            weight = None
        ff_out = self.ff_net(x2)

        return ff_out, weight


class GraphTransformerEncoder(nn.Module):
    def __init__(self, num_layers, nhead, emb_dim, ff_dim, dropout, set_beta):
        super(GraphTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.nhead = nhead
        self.hdim = emb_dim // nhead
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GraphTransformerLayer(emb_dim, nhead, ff_dim, dropout, set_beta)
            self.layers.append(layer)

    def forward(self, x, adj, knowledge, know_adj, requires_weight=False):
        weights_list = []
        bsz = x.shape[0]
        slen = x.shape[1]

        adj = adj.unsqueeze(1).expand(bsz, self.nhead, slen, slen)
        adj = adj.contiguous().view(bsz*self.nhead, slen, slen)

        know = torch.index_select(knowledge, 0, torch.flatten(know_adj)).contiguous().view(bsz, slen * slen, -1)

        x1 = x.transpose(0, 1)
        for i in range(self.num_layers):
            x1, weights = self.layers[i](x1, adj, know, requires_weight)
            if requires_weight:
                weights_list.append(weights)
        x1 = x1.transpose(0, 1)
        if requires_weight:
            return x1, weights_list
        else:
            return x1


class CskGraphTransformer(nn.Module):
    def __init__(self, model_size, utter_dim, conv_encoder, rnn_dropout, num_layers,
                 nhead, ff_dim, att_dropout, set_beta=True, emotion_emb=None, emotion_dim=0):
        super(CskGraphTransformer, self).__init__()
        self.utter_encoder = UtterEncoder2(model_size, utter_dim, conv_encoder, rnn_dropout)
        self.transformer_encoder = GraphTransformerEncoder(num_layers, nhead, utter_dim, ff_dim, att_dropout, set_beta)
        self.classifier = CausePredictor(utter_dim, utter_dim)

        self.csk_mapping = nn.Linear(1024, utter_dim)
        self.emotion_embeddings = nn.Embedding(emotion_emb.shape[0], emotion_emb.shape[1], padding_idx=0, _weight=emotion_emb)
        self.emotion_lin = nn.Linear(emotion_emb.shape[1], emotion_dim)
        self.emotion_mapping = nn.Linear(emotion_dim + utter_dim, utter_dim)

    def forward(self, conv_utterance, attention_mask, conv_len, adj, c_mask,
                emotion_label, knowledge_emb, know_adj, requires_weight=False):
        utter_emb = self.utter_encoder(conv_utterance, attention_mask, conv_len)
        # (num_know, bert_dim)
        knowledge = F.relu(self.csk_mapping(knowledge_emb))
        emo_emb = self.emotion_lin(self.emotion_embeddings(emotion_label))
        utter_emb = self.emotion_mapping(torch.cat([utter_emb, emo_emb], dim=-1))

        if requires_weight:
            output, weights = self.transformer_encoder(utter_emb, adj, knowledge, know_adj, requires_weight)
        else:
            output = self.transformer_encoder(utter_emb, adj, knowledge, know_adj, requires_weight)
            weights = None
        logits = self.classifier(output, c_mask)
        if requires_weight:
            return logits, weights
        else:
            return logits
