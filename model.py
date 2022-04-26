import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import RobertaModel


class UtterEncoder(nn.Module):
    def __init__(self, model_size, mapping_type, utter_dim, conv_encoder='none', rnn_dropout=None):
        super(UtterEncoder, self).__init__()
        encoder_path = 'roberta-' + model_size
        self.mapping_type = mapping_type
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
        # conv_utterance: [utter_num, max_len]
        output_data = self.encoder(conv_utterance, attention_mask=attention_mask)
        if self.mapping_type == 'cls':
            # [utter_num, token_dim]
            pooler_output = output_data.pooler_output
        elif self.mapping_type == 'max':
            # [utter_num, max_len, token_dim]
            last_hidden_state = output_data.last_hidden_state
            pooler_output = torch.max(last_hidden_state, dim=1)[0]
        elif self.mapping_type == 'mean':
            last_hidden_state = output_data.last_hidden_state
            pooler_output = torch.mean(last_hidden_state, dim=1)
        else:
            raise NotImplementedError()
        mapped_output = self.mapping(pooler_output)
        # [batch_size, [conv_size, utter_dim]]
        mapped_output_split = torch.split(mapped_output, conv_len)
        # [batch_size, conv_size, utter_dim]
        conv_output = pad_sequence(mapped_output_split, batch_first=True)
        if self.conv_encoder is not None:
            pad_conv = pack_padded_sequence(conv_output, conv_len, batch_first=True, enforce_sorted=False)
            pad_output = self.conv_encoder(pad_conv)[0]
            conv_output = pad_packed_sequence(pad_output, batch_first=True)[0]
        return conv_output


class DenseGraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(DenseGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_buffer('bias', None)
        self.reset_params()

    def reset_params(self):
        stdv = 1. / math.sqrt(self.weight.weight.size(1))
        self.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, mask=None, add_loop=True):
        # directed graph
        # x: [batch_size, num_node, node_dim];
        # adj: [batch_size, num_node, num_node];
        # mask: [batch_size, num_node]
        batch_size, num_node, _ = x.size()
        if add_loop:
            adj = adj.clone()
            eye = torch.eye(num_node, dtype=torch.float, device=adj.device).unsqueeze(0)
            # idx = torch.arange(num_node, dtype=torch.long, device=adj.device)
            # adj[:, idx, idx] = 1
            adj = adj + eye
        out = self.weight(x)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.contiguous().view(batch_size, num_node, 1).to(x.dtype)

        return out


class BiRelGraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(BiRelGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.self_loop_weight = nn.Linear(in_features, out_features, bias=False)
        self.self_weight = nn.Linear(in_features, out_features, bias=False)
        self.other_weight = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_buffer('bias', None)
        self.reset_params()

    def reset_params(self):
        stdv = 1. / math.sqrt(self.self_weight.weight.size(1))
        self.self_loop_weight.weight.data.uniform_(-stdv, stdv)
        self.self_weight.weight.data.uniform_(-stdv, stdv)
        self.other_weight.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_weight, adj_index, mask=None):
        # x: [batch_size, num_node, node_dim]
        # adj_index: [batch_size, 2, num_node, num_node]
        # adj_weight: [batch_size, num_node, num_node], adj_weight is normalized beforehand
        # mask: [batch_size, num_node]
        batch_size, num_node, _ = x.size()

        self_loop = self.self_loop_weight(x)
        self_effect = self.self_weight(x)
        other_effect = self.other_weight(x)
        self_item_adj = adj_index[:, 0, :, :].to(torch.float) * adj_weight
        other_item_adj = adj_index[:, 1, :, :].to(torch.float) * adj_weight
        self_effect = torch.matmul(self_item_adj, self_effect)
        other_effect = torch.matmul(other_item_adj, other_effect)
        out = (self_loop + self_effect + other_effect) / 2
        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.contiguous().view(batch_size, num_node, 1).to(x.dtype)

        return out


class Attention(nn.Module):
    def __init__(self, input_dim, add_activation):
        super(Attention, self).__init__()
        self.q = nn.Linear(input_dim, input_dim, bias=False)
        self.add_activation = add_activation

    def forward(self, query, key, mask):
        # query, key: [batch_size, seq_len, input_dim]
        # [batch_size, seq_len, seq_len]
        attentive_score = torch.matmul(key, self.q(query).transpose(1, 2))
        if self.add_activation:
            attentive_score = F.leaky_relu(attentive_score)
        # attentive_score = torch.masked_fill(attentive_score, mask.eq(0), -1e20)
        # attention_score = torch.softmax(attentive_score, dim=-1)
        attention_score = torch.sigmoid(attentive_score)
        attention_score = attention_score * mask
        return attention_score


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        linear_out = self.linear2(F.relu(self.linear1(x)))
        output = self.norm(linear_out + x)
        return output


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False, speaker_aware=False):
        super(GCNLayer, self).__init__()
        self.speaker_aware = speaker_aware
        if speaker_aware:
            self.gcn = BiRelGraphConv(in_features, out_features, bias)
        else:
            self.gcn = DenseGraphConv(in_features, out_features, bias)
        self.relu = nn.ReLU()
        self.mlp = MLP(out_features, out_features)

    def forward(self, x, adj=None, mask=None, add_loop=True, adj_weight=None, adj_index=None):
        if self.speaker_aware:
            out = self.gcn(x, adj_weight=adj_weight, adj_index=adj_index, mask=mask)
        else:
            out = self.gcn(x, adj=adj, mask=mask, add_loop=add_loop)
        out = self.mlp(self.relu(out))
        return out


class ERECEModel1(nn.Module):
    def __init__(self,
                 model_size,
                 mapping_type,
                 utter_dim,
                 bias,
                 num_emotions,
                 pruning_rate,
                 updating_rate,
                 hop,
                 non_emotion_index=0,
                 conv_encoder='none',
                 rnn_dropout=None,
                 speaker_aware=False,
                 add_activation=False):
        super(ERECEModel1, self).__init__()
        self.utter_encoder = UtterEncoder(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout)
        self.pruning_rate = pruning_rate
        self.updating_rate = updating_rate
        self.hop = hop
        self.speaker_aware = speaker_aware
        self.non_emotion_index = non_emotion_index
        self.er_layers = nn.ModuleList()
        self.er_classifiers = nn.ModuleList()
        self.ece_layers = nn.ModuleList()
        self.ece_classifiers = nn.ModuleList()
        for h in range(hop):
            er_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            ece_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            if h > 0:
                self.er_layers.append(er_layer)
            self.ece_layers.append(ece_layer)
            self.er_classifiers.append(nn.Linear(utter_dim, num_emotions))
            self.ece_classifiers.append(Attention(utter_dim, add_activation))

    def norm(self, adj):
        # directed graph
        deg_inv = (adj.sum(-1) + 1e-10).pow(-1)
        adj = adj * deg_inv.unsqueeze(2)
        return adj

    def forward(self,
                utterance,
                attention_mask,
                conv_len,
                adj=None,
                mask=None,
                adj_index=None,
                adj_weight=None):
        # adj, mask, adj_index: [batch_size, conv_len, conv_len]
        # adj, mask, adj_w, adj_i: x-axis denotes the target nodes, y-axis denotes the source nodes
        # utterance, attention_mask: [num_utt, utt_len]
        mask_for_node = mask.sum(1).clamp(max=1)
        all_er_predictions = []
        all_ece_predictions = []
        # utter_emb: [batch_size, conv_len, utt_dim]
        utter_emb = self.utter_encoder(utterance, attention_mask, conv_len)
        # [batch_size, conv_len, num_emotion]
        utter_prediction = self.er_classifiers[0](utter_emb)
        all_er_predictions.append(utter_prediction)
        # [batch_size, conv_len]
        non_emotion_utter_index = torch.argmax(utter_prediction.data, dim=-1).eq(self.non_emotion_index)
        non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                              non_emotion_utter_index, self.pruning_rate).to(utter_prediction.device)
        # [batch_size, conv_len, 1]
        non_emotion_utter = non_emotion_utter.unsqueeze(2).to(utter_prediction.device)
        dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(utter_prediction.device)
        non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
        # soft pruning. do not prune self-connected edges.
        # the adj matrix will be normalized
        if adj is not None:
            adj = adj * non_emotion_utter
            adj = self.norm(adj)
        if adj_weight is not None:
            adj_weight = adj_weight * non_emotion_utter
            adj_weight = self.norm(adj_weight)
        # iteration begin
        out = utter_emb
        for h in range(self.hop):
            if h == 0:
                if self.speaker_aware:
                    out = self.ece_layers[h](x=out, adj_weight=adj_weight, adj_index=adj_index, mask=mask_for_node)
                    pairing_score = self.ece_classifiers[0](out, out, mask)
                    all_ece_predictions.append(pairing_score)
                    adj_weight = adj_weight * (1 - self.updating_rate) + self.norm(pairing_score.data) * self.updating_rate
                else:
                    out = self.ece_layers[h](x=out, adj=adj, mask=mask_for_node, add_loop=True)
                    pairing_score = self.ece_classifiers[0](out, out, mask)
                    all_ece_predictions.append(pairing_score)
                    adj = adj * (1 - self.updating_rate) + self.norm(pairing_score.data) * self.updating_rate
            else:
                if self.speaker_aware:
                    out = self.er_layers[h-1](x=out, adj_weight=adj_weight, adj_index=adj_index, mask=mask_for_node)
                    utter_prediction = self.er_classifiers[h](out)
                    all_er_predictions.append(utter_prediction)
                    non_emotion_utter_index = torch.argmax(utter_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(utter_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2).to(utter_prediction.device)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(utter_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                    adj_weight = adj_weight * non_emotion_utter
                    adj_weight = self.norm(adj_weight)

                    out = self.ece_layers[h](x=out, adj_weight=adj_weight, adj_index=adj_index, mask=mask_for_node)
                    pairing_score = self.ece_classifiers[h](out, out, mask)
                    all_ece_predictions.append(pairing_score)
                    adj_weight = adj_weight * (1 - self.updating_rate) + self.norm(pairing_score.data) * self.updating_rate
                else:
                    out = self.er_layers[h-1](x=out, adj=adj, mask=mask_for_node, add_loop=True)
                    utter_prediction = self.er_classifiers[h](out)
                    all_er_predictions.append(utter_prediction)
                    non_emotion_utter_index = torch.argmax(utter_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(utter_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2).to(utter_prediction.device)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(utter_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                    adj = adj * non_emotion_utter
                    adj = self.norm(adj)

                    out = self.ece_layers[h](x=out, adj=adj, mask=mask_for_node, add_loop=True)
                    pairing_score = self.ece_classifiers[h](out, out, mask)
                    all_ece_predictions.append(pairing_score)
                    adj = adj * (1 - self.updating_rate) + self.norm(pairing_score.data) * self.updating_rate
        return all_er_predictions, all_ece_predictions


class ERECEModel2(nn.Module):
    def __init__(self,
                 model_size,
                 mapping_type,
                 utter_dim,
                 bias,
                 num_emotions,
                 pruning_rate,
                 updating_rate,
                 hop,
                 non_emotion_index=0,
                 conv_encoder='none',
                 rnn_dropout=None,
                 speaker_aware=False,
                 add_activation=False):
        super(ERECEModel2, self).__init__()
        self.utter_encoder = UtterEncoder(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout)
        self.pruning_rate = pruning_rate
        self.updating_rate = updating_rate
        self.hop = hop
        self.speaker_aware = speaker_aware
        self.non_emotion_index = non_emotion_index
        self.er_layers = nn.ModuleList()
        self.er_classifiers = nn.ModuleList()
        self.ece_layers = nn.ModuleList()
        self.ece_classifiers = nn.ModuleList()
        for h in range(hop):
            er_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            ece_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            if h > 0:
                self.ece_layers.append(ece_layer)
            self.er_layers.append(er_layer)
            self.er_classifiers.append(nn.Linear(utter_dim, num_emotions))
            self.ece_classifiers.append(Attention(utter_dim, add_activation))

    def norm(self, adj):
        # directed graph
        deg_inv = (adj.sum(-1) + 1e-10).pow(-1)
        adj = adj * deg_inv.unsqueeze(2)
        return adj

    def forward(self,
                utterance,
                attention_mask,
                conv_len,
                mask=None,
                adj_index=None
                ):
        all_er_predictions = []
        all_ece_predictions = []
        mask_for_node = mask.sum(-1).clamp(max=1)
        utter_emb = self.utter_encoder(utterance, attention_mask, conv_len)
        # [batch_size, conv_len, conv_len]
        pairing_score = self.ece_classifiers[0](utter_emb, utter_emb, mask)
        all_ece_predictions.append(pairing_score)

        adj = self.norm(pairing_score.data)

        # iteration begin
        out = utter_emb
        for h in range(self.hop):
            if h == 0:
                if self.speaker_aware:
                    out = self.er_layers[h](x=out, adj_weight=adj, adj_index=adj_index, mask=mask_for_node)
                    utter_prediction = self.er_classifiers[h](out)
                    all_er_predictions.append(utter_prediction)
                    non_emotion_utter_index = torch.argmax(utter_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(utter_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2).to(utter_prediction.device)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(utter_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                    adj = adj * non_emotion_utter
                    adj = self.norm(adj)
                else:
                    out = self.er_layers[h](x=out, adj=adj, mask=mask_for_node, add_loop=True)
                    utter_prediction = self.er_classifiers[h](out)
                    all_er_predictions.append(utter_prediction)
                    non_emotion_utter_index = torch.argmax(utter_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(utter_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2).to(utter_prediction.device)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(utter_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                    adj = adj * non_emotion_utter
                    adj = self.norm(adj)
            else:
                if self.speaker_aware:
                    out = self.ece_layers[h-1](x=out, adj_weight=adj, adj_index=adj_index, mask=mask_for_node)
                    pairing_score = self.ece_classifiers[h](out, out, mask)
                    all_ece_predictions.append(pairing_score)
                    adj = adj * (1 - self.updating_rate) + self.norm(pairing_score.data) * self.updating_rate

                    out = self.er_layers[h](x=out, adj_weight=adj, adj_index=adj_index, mask=mask_for_node)
                    utter_prediction = self.er_classifiers[h](out)
                    all_er_predictions.append(utter_prediction)
                    non_emotion_utter_index = torch.argmax(utter_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(utter_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2).to(utter_prediction.device)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(utter_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                    adj = adj * non_emotion_utter
                    adj = self.norm(adj)
                else:
                    out = self.ece_layers[h-1](x=out, adj=adj, mask=mask_for_node, add_loop=True)
                    pairing_score = self.ece_classifiers[h](out, out, mask)
                    all_ece_predictions.append(pairing_score)
                    adj = adj * (1 - self.updating_rate) + self.norm(pairing_score.data) * self.updating_rate

                    out = self.er_layers[h](x=out, adj=adj, mask=mask_for_node, add_loop=True)
                    utter_prediction = self.er_classifiers[h](out)
                    all_er_predictions.append(utter_prediction)
                    non_emotion_utter_index = torch.argmax(utter_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(utter_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2).to(utter_prediction.device)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(utter_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)

                    adj = adj * non_emotion_utter
                    adj = self.norm(adj)

        return all_er_predictions, all_ece_predictions


class ERECEModel3(nn.Module):
    def __init__(self,
                 model_size,
                 mapping_type,
                 utter_dim,
                 bias,
                 num_emotions,
                 pruning_rate,
                 updating_rate,
                 hop,
                 non_emotion_index=0,
                 conv_encoder='none',
                 rnn_dropout=None,
                 speaker_aware=False,
                 add_activation=False):
        super(ERECEModel3, self).__init__()
        self.utter_encoder = UtterEncoder(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout)
        self.pruning_rate = pruning_rate
        self.updating_rate = updating_rate
        self.hop = hop
        self.speaker_aware = speaker_aware
        self.non_emotion_index = non_emotion_index
        self.ece_linear = nn.Linear(utter_dim, utter_dim)
        self.er_linear = nn.Linear(utter_dim, utter_dim)

        self.er_layers_1st_stream = nn.ModuleList()
        self.er_classifiers_1st_stream = nn.ModuleList()
        self.ece_layers_1st_stream = nn.ModuleList()
        self.ece_classifiers_1st_stream = nn.ModuleList()

        self.er_layers_2nd_stream = nn.ModuleList()
        self.er_classifiers_2nd_stream = nn.ModuleList()
        self.ece_layers_2nd_stream = nn.ModuleList()
        self.ece_classifiers_2nd_stream = nn.ModuleList()

        for h in range(hop):
            er_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            ece_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            if h > 0:
                self.ece_layers_1st_stream.append(ece_layer)
            self.er_layers_1st_stream.append(er_layer)
            self.er_classifiers_1st_stream.append(nn.Linear(utter_dim, num_emotions))
            self.ece_classifiers_1st_stream.append(Attention(utter_dim, add_activation))

            er_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            ece_layer = GCNLayer(utter_dim, utter_dim, bias, speaker_aware)
            if h > 0:
                self.er_layers_2nd_stream.append(er_layer)
            self.ece_layers_2nd_stream.append(ece_layer)
            self.er_classifiers_2nd_stream.append(nn.Linear(utter_dim, num_emotions))
            self.ece_classifiers_2nd_stream.append(Attention(utter_dim, add_activation))

    def norm(self, adj):
        # directed graph
        deg_inv = (adj.sum(-1) + 1e-10).pow(-1)
        adj = adj * deg_inv.unsqueeze(2)
        return adj

    def forward(self,
                utterance,
                attention_mask,
                conv_len,
                mask=None,
                adj_index=None,
                adj_weight=None):
        mask_for_node = mask.sum(-1).clamp(max=1)
        all_er_prediction_1st_stream = []
        all_ece_prediction_1st_stream = []
        all_er_prediction_2nd_stream = []
        all_ece_prediction_2nd_stream = []
        utter_emb = self.utter_encoder(utterance, attention_mask, conv_len)
        stream_1st_emb = self.ece_linear(utter_emb)
        stream_2nd_emb = self.er_linear(utter_emb)

        stream_1st_pair_score = self.ece_classifiers_1st_stream[0](stream_1st_emb, stream_1st_emb, mask)
        all_ece_prediction_1st_stream.append(stream_1st_pair_score)
        stream_1st_ece_adj = stream_1st_pair_score.data

        stream_2nd_er_prediction = self.er_classifiers_2nd_stream[0](stream_2nd_emb)
        all_er_prediction_2nd_stream.append(stream_2nd_er_prediction)
        # [batch_size, conv_len]
        non_emotion_utter_index = torch.argmax(stream_2nd_er_prediction.data, dim=-1).eq(self.non_emotion_index)
        non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                              non_emotion_utter_index, self.pruning_rate).to(stream_2nd_er_prediction.device)
        non_emotion_utter = non_emotion_utter.unsqueeze(2)
        dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(stream_2nd_er_prediction.device)
        non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
        # soft pruning.
        stream_1st_ece_adj = self.norm(stream_1st_ece_adj)
        stream_2nd_er_adj = adj_weight * non_emotion_utter
        stream_2nd_er_adj = self.norm(stream_2nd_er_adj)

        out_1st = stream_1st_emb
        out_2nd = stream_2nd_emb
        for h in range(self.hop):
            if h == 0:
                if self.speaker_aware:
                    # 1st stream for er; 2nd stream for ece
                    out_1st = self.er_layers_1st_stream[h](x=out_1st, adj_weight=stream_1st_ece_adj,
                                                           adj_index=adj_index, mask=mask_for_node)
                    stream_1st_er_prediction = self.er_classifiers_1st_stream[h](out_1st)
                    all_er_prediction_1st_stream.append(stream_1st_er_prediction)
                    non_emotion_utter_index = torch.argmax(stream_1st_er_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(stream_1st_er_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(stream_1st_er_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                    stream_1st_er_adj = stream_1st_ece_adj * non_emotion_utter
                    stream_1st_er_adj = self.norm(stream_1st_er_adj)

                    out_2nd = self.ece_layers_2nd_stream[h](x=out_2nd, adj_weight=stream_2nd_er_adj,
                                                            adj_index=adj_index, mask=mask_for_node)
                    stream_2nd_ece_prediction = self.ece_classifiers_2nd_stream[h](out_2nd, out_2nd, mask)
                    all_ece_prediction_2nd_stream.append(stream_2nd_ece_prediction)
                    stream_2nd_ece_adj = stream_2nd_er_adj * (1 - self.updating_rate) + self.norm(stream_2nd_ece_prediction.data) * self.updating_rate

                    updating_adj = (stream_1st_er_adj + stream_2nd_ece_adj) / 2
                else:
                    out_1st = self.er_layers_1st_stream[h](x=out_1st, adj=stream_1st_ece_adj, mask=mask_for_node, add_loop=True)
                    stream_1st_er_prediction = self.er_classifiers_1st_stream[h](out_1st)
                    all_er_prediction_1st_stream.append(stream_1st_er_prediction)
                    non_emotion_utter_index = torch.argmax(stream_1st_er_prediction.data, dim=-1).eq(self.non_emotion_index)
                    non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                          non_emotion_utter_index, self.pruning_rate).to(stream_1st_er_prediction.device)
                    non_emotion_utter = non_emotion_utter.unsqueeze(2)
                    dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(stream_1st_er_prediction.device)
                    non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                    stream_1st_er_adj = stream_1st_ece_adj * non_emotion_utter
                    stream_1st_er_adj = self.norm(stream_1st_er_adj)

                    out_2nd = self.ece_layers_2nd_stream[h](x=out_2nd, adj=stream_2nd_er_adj, mask=mask_for_node, add_loop=True)
                    stream_2nd_ece_prediction = self.ece_classifiers_2nd_stream[h](out_2nd, out_2nd, mask)
                    all_ece_prediction_2nd_stream.append(stream_2nd_ece_prediction)
                    stream_2nd_ece_adj = stream_2nd_er_adj * (1 - self.updating_rate) + self.norm(stream_2nd_ece_prediction.data) * self.updating_rate

                    updating_adj = (stream_1st_er_adj + stream_2nd_ece_adj) / 2
            else:
                if self.speaker_aware:
                    out_1st = self.ece_layers_1st_stream[h-1](x=out_1st, adj_weight=updating_adj,
                                                              adj_index=adj_index, mask=mask_for_node)
                else:
                    out_1st = self.ece_layers_1st_stream[h-1](x=out_1st, adj=updating_adj, mask=mask_for_node, add_loop=True)
                stream_1st_ece_prediction = self.ece_classifiers_1st_stream[h](out_1st, out_1st, mask)
                all_ece_prediction_1st_stream.append(stream_1st_ece_prediction)
                stream_1st_ece_adj = updating_adj * (1 - self.updating_rate) + self.norm(stream_1st_ece_prediction.data) * self.updating_rate

                if self.speaker_aware:
                    out_1st = self.er_layers_1st_stream[h](x=out_1st, adj_weight=stream_1st_ece_adj,
                                                           adj_index=adj_index, mask=mask_for_node)
                else:
                    out_1st = self.er_layers_1st_stream[h](x=out_1st, adj=stream_1st_ece_adj, mask=mask_for_node, add_loop=True)
                stream_1st_er_prediction = self.er_classifiers_1st_stream[h](out_1st)
                all_er_prediction_1st_stream.append(stream_1st_er_prediction)
                non_emotion_utter_index = torch.argmax(stream_1st_er_prediction.data, dim=-1).eq(self.non_emotion_index)
                non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                      non_emotion_utter_index, self.pruning_rate).to(stream_1st_er_prediction.device)
                non_emotion_utter = non_emotion_utter.unsqueeze(2)
                dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(stream_1st_er_prediction.device)
                non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                stream_1st_er_adj = stream_1st_ece_adj * non_emotion_utter
                stream_1st_er_adj = self.norm(stream_1st_er_adj)

                if self.speaker_aware:
                    out_2nd = self.er_layers_2nd_stream[h-1](x=out_2nd, adj_weight=updating_adj,
                                                             adj_index=adj_index, mask=mask_for_node)
                else:
                    out_2nd = self.er_layers_2nd_stream[h-1](x=out_2nd, adj=updating_adj, mask=mask_for_node, add_loop=True)
                stream_2nd_er_prediction = self.er_classifiers_2nd_stream[h](out_2nd)
                all_er_prediction_2nd_stream.append(stream_2nd_er_prediction)
                non_emotion_utter_index = torch.argmax(stream_2nd_er_prediction.data, dim=-1).eq(self.non_emotion_index)
                non_emotion_utter = torch.masked_fill(torch.ones_like(non_emotion_utter_index),
                                                      non_emotion_utter_index, self.pruning_rate).to(stream_2nd_er_prediction.device)
                non_emotion_utter = non_emotion_utter.unsqueeze(2)
                dig = torch.eye(non_emotion_utter.shape[1]).unsqueeze(0).to(stream_2nd_er_prediction.device)
                non_emotion_utter = torch.clamp(non_emotion_utter + dig, max=1)
                stream_2nd_er_adj = updating_adj * non_emotion_utter
                stream_2nd_er_adj = self.norm(stream_2nd_er_adj)

                if self.speaker_aware:
                    out_2nd = self.ece_layers_2nd_stream[h](x=out_2nd, adj_weight=stream_2nd_er_adj,
                                                            adj_index=adj_index, mask=mask_for_node)
                else:
                    out_2nd = self.ece_layers_2nd_stream[h](x=out_2nd, adj=stream_2nd_er_adj, mask=mask_for_node, add_loop=True)
                stream_2nd_ece_prediction = self.ece_classifiers_2nd_stream[h](out_2nd, out_2nd, mask)
                all_ece_prediction_2nd_stream.append(stream_2nd_ece_prediction)
                stream_2nd_ece_adj = stream_2nd_er_adj * (1 - self.updating_rate) + self.norm(stream_2nd_ece_prediction.data) * self.updating_rate
                updating_adj = (stream_1st_er_adj + stream_2nd_ece_adj) / 2

        return [all_er_prediction_1st_stream, all_er_prediction_2nd_stream], \
               [all_ece_prediction_1st_stream, all_ece_prediction_2nd_stream]


class GAT(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout, alpha, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        assert out_features == in_features // num_heads
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(1, in_features, in_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(1, num_heads, 2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, adj_weight=None):
        # h: [batch_size, conv_len, in_features]
        # adj: [batch_size, conv_len, conv_len]
        # adj_weight: [batch_size, conv_len, conv_len]
        batch_size, conv_len, _ = h.shape
        mask = torch.eq(adj, 0).unsqueeze(1).expand(batch_size, self.num_heads, conv_len, conv_len)
        # Wh: [batch_size, conv_len, in_features] -> [batch_size, head, conv_len, out_features]
        Wh = torch.matmul(h, self.W)
        Wh = Wh.contiguous().view(batch_size, conv_len, self.num_heads, self.out_features).transpose(1, 2)

        # [batch_size, head, conv_len, out_features] * [1, head, out_features, 1] ->
        # [batch_size, head, conv_len, 1]
        Wh1 = torch.matmul(Wh, self.a[:, :, :self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[:, :, self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2)
        # [batch_size, head, conv_len, conv_len]
        attention = torch.masked_fill(e, mask, -9e15)
        attention = F.softmax(attention, dim=-1)
        if adj_weight is not None:
            # [batch_size, 1, conv_len, in_features]
            adj_weight = adj_weight.unsqueeze(1)
            attention = attention * adj_weight
        attention = F.dropout(attention, self.dropout, training=self.training)
        # [batch_size, head, conv_len, conv_len] * [batch_size, head, conv_len, out_features] ->
        # [batch_size, head, conv_len, out_features] -> [batch_size, conv_len, in_features]
        h_prime = torch.matmul(attention, Wh)
        h_prime = h_prime.transpose(1, 2).contiguous().view(batch_size, conv_len, self.in_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class BiGAT(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout, alpha, concat=True, fusion_type='sum'):
        super(BiGAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        assert out_features == in_features // num_heads
        self.alpha = alpha
        self.concat = concat
        self.fusion_type = fusion_type

        self.self_gat = GAT(in_features, out_features, num_heads, dropout, alpha, concat)
        self.other_gat = GAT(in_features, out_features, num_heads, dropout, alpha, concat)

        if fusion_type == 'cat':
            self.linear_map = nn.Linear(2*in_features, in_features)
        elif fusion_type == 'att':
            raise NotImplementedError()
        else:
            pass

    def forward(self, h, adj_self, adj_other, adj_weight_self=None, adj_weight_other=None):
        self_out = self.self_gat(h, adj_self, adj_weight_self)
        other_out = self.other_gat(h, adj_other, adj_weight_other)
        if self.fusion_type == 'cat':
            out = self.linear_map(torch.cat([self_out, other_out], dim=-1))
        elif self.fusion_type == 'att':
            raise NotImplementedError()
        else:
            out = self_out + other_out
        return out


def norm(adj):
    # directed graph
    deg_inv = (adj.sum(-1) + 1e-10).pow(-1)
    adj = adj * deg_inv.unsqueeze(2)
    return adj


class TransformerConv(nn.Module):
    def __init__(self, nhead, emb_dim, dropout, set_beta=True):
        super(TransformerConv, self).__init__()
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.set_beta = set_beta
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        if set_beta:
            self.skip_lin = nn.Linear(emb_dim, emb_dim, bias=False)
            self.beta = nn.Linear(3*emb_dim, 1, bias=False)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.q_proj_weight)
        torch.nn.init.xavier_uniform_(self.k_proj_weight)
        torch.nn.init.xavier_uniform_(self.v_proj_weight)
        if self.set_beta:
            torch.nn.init.xavier_uniform_(self.skip_lin.weight)
            torch.nn.init.xavier_uniform_(self.beta.weight)

    def forward(self, x, mask=None, weight_mask=None, require_weight=False):
        # input size: (slen, bsz, nh*hdim)
        slen = x.size(0)
        bsz = x.size(1)

        scaling = float(self.head_dim) ** -0.5
        query = F.linear(x, self.q_proj_weight)
        key = F.linear(x, self.k_proj_weight)
        value = F.linear(x, self.v_proj_weight)

        # (slen, bsz, nh*hdim) -> (slen, bsz*nh, hdim) -> (bsz*nh, slen, hdim)
        query = query.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1)
        # (bsz*nh, slen, slen)
        attn_weight = torch.matmul(query, key.transpose(1, 2))
        attn_weight = attn_weight * scaling
        if mask is not None:
            mask = mask.unsqueeze(1).expand(bsz, self.nhead, slen, slen).contiguous().view(bsz * self.nhead, slen, slen)
            attn_weight = torch.masked_fill(attn_weight, mask.eq(0), -9e15)
            attn_weight = F.softmax(attn_weight, dim=-1) * mask
        if weight_mask is not None:
            weight_mask = weight_mask.unsqueeze(1).expand(bsz, self.nhead, slen, slen).contiguous().view(bsz * self.nhead, slen, slen)
            attn_weight = torch.masked_fill(attn_weight, weight_mask.eq(0), -9e16)
            attn_weight = F.softmax(attn_weight, dim=-1) * weight_mask
            attn_weight = norm(attn_weight)
        attn_score = F.dropout(attn_weight, p=self.dropout, training=self.training)
        # (bsz*nh, slen, slen) * (bsz*nh, slen, hdim) -> (bsz*nh, slen, hdim)
        attn_output = torch.matmul(attn_score, value)
        # (bsz*nh, slen, hdim) -> (slen, bsz*nh, hdim) -> (slen, bsz, nh*hdim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(slen, bsz, -1)

        if self.set_beta:
            r = self.skip_lin(x)
            # (slen, bsz, 3*nh*hdim) -> (slen, bsz, 1)
            beta_gate = self.beta(torch.cat([attn_output, r, attn_output - r], dim=-1))
            beta_gate = F.sigmoid(beta_gate)
            output = beta_gate * r + (1 - beta_gate) * attn_output
        else:
            output = attn_output + x

        if require_weight:
            return output, attn_score
        else:
            return output


class CausePredictor(nn.Module):
    def __init__(self, input_dim, mlp_dim, mlp_dropout=0.1):
        super(CausePredictor, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.mlp = nn.Sequential(nn.Linear(2*input_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout),
                                 nn.Linear(mlp_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.predictor_weight = nn.Linear(mlp_dim, 1, False)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        conv_len = x.shape[1]
        x_dim = x.shape[2]
        x_source = x.unsqueeze(1).expand(batch_size, conv_len, conv_len, x_dim)
        x_target = x.unsqueeze(2).expand(batch_size, conv_len, conv_len, x_dim)
        # [batch_size, conv_len, conv_len, 2*x_dim]
        x_cat = torch.cat([x_source, x_target], dim=-1)
        # [batch_size, conv_len, conv_len]
        predict_score = self.predictor_weight(self.mlp(x_cat)).squeeze(-1)
        predict_score = torch.sigmoid(predict_score) * mask

        return predict_score


# DAG is taken from DAG-ERC

class DAGNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, pooler_type='all'):
        super(DAGNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.pooler_type = pooler_type
        for i in range(self.num_layers):
            self.layers.append(GRNLayer(input_size, hidden_size))
        if self.pooler_type == 'all':
            self.emb_lin = nn.Linear((num_layers+1)*hidden_size, hidden_size)
        else:
            self.emb_lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, features, adj, s_mask, o_mask):
        H = [features]
        H1 = features
        for i in range(self.num_layers):
            H1 = self.dropout(self.layers[i](H1, adj, s_mask, o_mask))
            H.append(H1)
        if self.pooler_type == 'all':
            H = torch.cat(H, dim=2)
            utter_emb = self.emb_lin(H)
        else:
            utter_emb = self.emb_lin(H[-1])
        return utter_emb


class GRNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRNLayer, self).__init__()
        self.gru_c = nn.GRUCell(input_size, hidden_size)
        self.gru_p = nn.GRUCell(input_size, hidden_size)
        self.gat = GraphAttention(hidden_size)

    def forward(self, features, adj, s_mask, o_mask):
        # features: [batch_size, num_utter, utter_dim]
        num_utter = features.size()[1]
        # the first utterance
        # [batch_size, 1, utter_dim]
        C = self.gru_c(features[:, 0, :]).unsqueeze(1)
        M = torch.zeros_like(C).squeeze(1)
        P = self.gru_p(M, features[:, 0, :]).unsqueeze(1)
        H1 = C + P
        for i in range(1, num_utter):
            _, M = self.gat(features[:, i, :], H1, H1, adj[:, i, :i], s_mask[:, i, :i], o_mask[:, i, :i])
            C = self.gru_c(features[:, i, :], M).unsqueeze(1)
            P = self.gru_p(M, features[:, i, :]).unsqueeze(1)
            H_temp = C + P
            # [batch_size, i+1, utter_dim]
            H1 = torch.cat((H1, H_temp), dim=1)
        return H1


def mask_logic(alpha, adj):
    return alpha - (1 - adj) * 1e30


class GraphAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GraphAttention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, Q, K, V, adj, s_mask, o_mask):
        B = K.size()[0]
        N = K.size()[1]
        Q = Q.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)；
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
