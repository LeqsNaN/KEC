import torch
import pickle
from dataset import BaseDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np


class SkaigDataset(Dataset):
    def __init__(self, model_size='base', phase='train', window=10):
        super(SkaigDataset, self).__init__()
        self.base_dataset = BaseDataset(model_size, phase)
        self.window = window * 2
        knowledge_path = 'skaig_data/dailydialog_' + phase + '_know_processed.pkl'
        data = pickle.load(open(knowledge_path, 'rb'), encoding='latin1')
        knowledge, knowledge_adj = data[0], data[1]
        self.knowledge = knowledge
        self.knowledge_adj = knowledge_adj

    def __getitem__(self, item):
        utter, token_id, att_mask, label, adj_idx, evid, msk, clen, act = self.base_dataset[item]
        edge_mask = msk.clone()
        knowledge = torch.tensor(self.knowledge[item], dtype=torch.float)
        knowledge_adj = torch.tensor(self.knowledge_adj[item], dtype=torch.long)
        knowledge_adj = knowledge_adj.triu(-self.window)
        edge_mask = edge_mask.triu(-self.window)
        adj_idx = adj_idx.triu(-self.window)
        return utter, token_id, att_mask, label, adj_idx, evid, msk, \
               clen, act, knowledge, knowledge_adj, edge_mask

    def __len__(self):
        return len(self.base_dataset)


def collate_fn_know(data):
    token_ids_1 = []
    attention_mask_1 = []
    label = []
    adj_index = []
    ece_pair = []
    mask = []
    edge_mask = []
    clen = []
    act_label = []
    knowledge = []
    knowledge_adj = []
    for i, d in enumerate(data):
        token_ids_1.append(d[1])
        attention_mask_1.append(d[2])
        label.append(d[3])
        # [2, conv_len, conv_len]
        adj_index.append(d[4])
        ece_pair.append(d[5])
        mask.append(d[6])
        clen.append(d[7])
        act_label.append(d[8])
        knowledge.append(d[9])
        knowledge_adj.append(d[10])
        edge_mask.append(d[11])
    label = pad_sequence(label, batch_first=True, padding_value=-1)
    act_label = pad_sequence(act_label, batch_first=True, padding_value=-1)
    max_len = max(clen)
    mask = [torch.cat([torch.cat([m, torch.zeros(max_len - m.shape[0], m.shape[1])], dim=0), torch.zeros(max_len, max_len - m.shape[1])], dim=1) for m in mask]
    mask = torch.stack(mask, dim=0)
    edge_mask = [torch.cat([torch.cat([m, torch.zeros(max_len - m.shape[0], m.shape[1])], dim=0), torch.zeros(max_len, max_len - m.shape[1])], dim=1) for m in edge_mask]
    edge_mask = torch.stack(edge_mask, dim=0)
    ece_pair = [torch.cat([torch.cat([ep, torch.zeros(max_len - ep.shape[0], ep.shape[1])], dim=0), torch.zeros(max_len, max_len - ep.shape[1])], dim=1) for ep in ece_pair]
    ece_pair = torch.stack(ece_pair, dim=0)
    adj_index = [torch.cat([torch.cat([a, torch.zeros(2, max_len - a.shape[1], a.shape[2])], dim=1), torch.zeros(2, max_len, max_len - a.shape[2])], dim=2) for a in adj_index]
    # [batch_size, 2, conv_len, conv_len]
    adj_index = torch.stack(adj_index, dim=0)
    num_knowledge = [k.shape[0] for k in knowledge]
    knowledge = torch.cat(knowledge, dim=0)
    accu_know = 0
    for i in range(len(num_knowledge)):
        knowledge_adj[i] = knowledge_adj[i] + accu_know
        accu_know += num_knowledge[i]
        knowledge_adj[i] = torch.cat([torch.cat([knowledge_adj[i], torch.zeros(max_len - knowledge_adj[i].shape[0], knowledge_adj[i].shape[1])], dim=0), torch.zeros(max_len, max_len-knowledge_adj[i].shape[1])], dim=1)
    knowledge_adj = torch.stack(knowledge_adj, dim=0)

    return token_ids_1, attention_mask_1, clen, mask, edge_mask, adj_index, \
           label, ece_pair, act_label, knowledge, knowledge_adj, num_knowledge


def get_dataloaders(model_size, batch_size, valid_shuffle, window=10):
    train_set = SkaigDataset(model_size, 'train', window)
    dev_set = SkaigDataset(model_size, 'dev', window)
    test_set = SkaigDataset(model_size, 'test', window)
    train_loader = DataLoader(train_set, batch_size, True, collate_fn=collate_fn_know)
    dev_loader = DataLoader(dev_set, batch_size, valid_shuffle, collate_fn=collate_fn_know)
    test_loader = DataLoader(test_set, batch_size, valid_shuffle, collate_fn=collate_fn_know)

    return train_loader, dev_loader, test_loader


class KagDataset(Dataset):
    def __init__(self, model_size='base', phase='train', unidirection=False):
        super(KagDataset, self).__init__()
        self.base_dataset = BaseDataset(model_size, phase)
        self.unidirection = unidirection
        know_adj_path = 'kag_data/' + phase + '_conceptnet_adj_1.pkl'
        know_adj_processed_path = 'kag_data/' + phase + '_conceptnet_adj_processed.pkl'
        if os.path.exists(know_adj_processed_path):
            know_data = pickle.load(open(know_adj_processed_path, 'rb'), encoding='latin1')
            self.knowledge_text = know_data[0]
            self.knowledge_attention_mask = know_data[1]
            self.knowledge_adj = know_data[2]
            self.knowledge_len = know_data[3]
            self.sequence_adj = know_data[4]
        else:
            self.knowledge_text = []
            self.knowledge_attention_mask = []
            self.knowledge_adj = []
            self.knowledge_len = []
            self.sequence_adj = []
            know_data = pickle.load(open(know_adj_path, 'rb'), encoding='latin1')
            for conv_know, label in zip(know_data, self.base_dataset.emotion):
                num_utter = len(label)
                conv_adj = np.zeros((num_utter, num_utter), dtype=np.int)
                know_len = []
                kw_text = []
                kw_att_mask = []
                index_count = 1
                for k, v in conv_know.items():
                    source_node = int(k.split('-')[0])
                    target_node = int(k.split('-')[1])
                    conv_adj[target_node, source_node] = index_count
                    index_count += 1
                    know_len.append(len(v))
                    for kw in v:
                        processed_kw = self.base_dataset.tokenizer(kw)
                        kw_text.append(torch.tensor(processed_kw.input_ids, dtype=torch.long))
                        kw_att_mask.append(torch.tensor(processed_kw.attention_mask))
                kw_text = pad_sequence(kw_text, batch_first=True, padding_value=1)
                kw_att_mask = pad_sequence(kw_att_mask, batch_first=True, padding_value=0)
                conv_adj = torch.tensor(conv_adj, dtype=torch.long)
                seq_adj = torch.zeros_like(conv_adj, dtype=torch.long)
                for i in range(num_utter):
                    if i == 0:
                        seq_adj[i+1, i] = 1
                    elif 0 < i < num_utter - 1:
                        seq_adj[i+1, i] = 1
                        seq_adj[i-1, i] = 1
                    else:
                        seq_adj[i-1, i] = 1
                self.knowledge_text.append(kw_text)
                self.knowledge_attention_mask.append(kw_att_mask)
                self.knowledge_adj.append(conv_adj)
                self.knowledge_len.append(know_len)
                self.sequence_adj.append(seq_adj)
            pickle.dump([self.knowledge_text, self.knowledge_attention_mask, self.knowledge_adj,
                         self.knowledge_len, self.sequence_adj], open(know_adj_processed_path, 'wb'))

    def __getitem__(self, item):
        utter, token_id, att_mask, label, adj_idx, evid, msk, clen, act = self.base_dataset[item]
        knowledge = self.knowledge_text[item]
        know_attn_mask = self.knowledge_attention_mask[item]
        know_adj = self.knowledge_adj[item]
        seq_adj = self.sequence_adj[item]
        if self.unidirection:
            seq_adj = torch.tril(seq_adj, 0)
        know_len = self.knowledge_len[item]
        return token_id, att_mask, label, evid, msk, clen, \
               knowledge, know_attn_mask, know_adj, seq_adj, know_len

    def __len__(self):
        return len(self.base_dataset.emotion)


def collate_fn_kag(data):
    token_ids = []
    attention_mask = []
    label = []
    ece_pair = []
    mask = []
    clen = []
    knowledge = []
    knowledge_attention_mask = []
    knowledge_adj = []
    sequence_adj = []
    knowledge_len = []
    batch_knowledge_len = []
    for d in data:
        tk_ids = d[0]
        at_msk = d[1]
        lbl = d[2]
        ece_p = d[3]
        msk = d[4]
        cl = d[5]
        know = d[6]
        know_att_msk = d[7]
        know_adj = d[8]
        seq_adj = d[9]
        know_len = d[10]
        token_ids.append(tk_ids)
        attention_mask.append(at_msk)
        label.append(lbl)
        ece_pair.append(ece_p)
        mask.append(msk)
        clen.append(cl)
        knowledge.append(know)
        knowledge_adj.append(know_adj)
        knowledge_attention_mask.append(know_att_msk)
        sequence_adj.append(seq_adj)
        knowledge_len.append(know_len)
        batch_knowledge_len.append(sum(know_len))

    label = pad_sequence(label, batch_first=True, padding_value=-1)
    max_len = max(clen)
    mask = [torch.cat([torch.cat([m, torch.zeros(max_len - m.shape[0], m.shape[1])], dim=0), torch.zeros(max_len, max_len - m.shape[1])], dim=1) for m in mask]
    mask = torch.stack(mask, dim=0)
    ece_pair = [torch.cat([torch.cat([ep, torch.zeros(max_len - ep.shape[0], ep.shape[1])], dim=0), torch.zeros(max_len, max_len - ep.shape[1])], dim=1) for ep in ece_pair]
    ece_pair = torch.stack(ece_pair, dim=0)
    sequence_adj = [torch.cat([torch.cat([sa, torch.zeros(max_len - sa.shape[0], sa.shape[1])], dim=0), torch.zeros(max_len, max_len - sa.shape[1])], dim=1) for sa in sequence_adj]
    sequence_adj = torch.stack(sequence_adj, dim=0)

    max_knowledge_len = max([k.shape[1] if k is not None else 0 for k in knowledge])
    num_knowledge = [len(k) for k in knowledge_len]
    knowledge_ = []
    knowledge_attention_mask_ = []
    for k, km in zip(knowledge, knowledge_attention_mask):
        knowledge_.append(torch.cat([k, torch.ones(k.shape[0], max_knowledge_len - k.shape[1])], dim=1))
        knowledge_attention_mask_.append(torch.cat([km, torch.zeros(km.shape[0], max_knowledge_len - km.shape[1])], dim=1))
    knowledge = torch.cat(knowledge_, dim=0)
    knowledge_attention_mask = torch.cat(knowledge_attention_mask_, dim=0)

    accu_know = 0
    for i in range(len(num_knowledge)):
        mmask = knowledge_adj[i].eq(0)
        knowledge_adj[i] = knowledge_adj[i] + accu_know
        knowledge_adj[i] = torch.masked_fill(knowledge_adj[i], mmask, 0)
        accu_know += num_knowledge[i]
        knowledge_adj[i] = torch.cat([torch.cat([knowledge_adj[i], torch.zeros(max_len - knowledge_adj[i].shape[0], knowledge_adj[i].shape[1])], dim=0), torch.zeros(max_len, max_len - knowledge_adj[i].shape[1])], dim=1)
    knowledge_adj = torch.stack(knowledge_adj, dim=0)

    return token_ids, attention_mask, clen, \
           label, mask, ece_pair, knowledge, knowledge_attention_mask, \
           knowledge_adj, sequence_adj, knowledge_len, batch_knowledge_len


def get_kag_dataloaders(model_size, batch_size, valid_shuffle, unidirection=False):
    train_set = KagDataset(model_size, 'train', unidirection)
    dev_set = KagDataset(model_size, 'dev', unidirection)
    test_set = KagDataset(model_size, 'test', unidirection)
    train_loader = DataLoader(train_set, batch_size, True, collate_fn=collate_fn_kag)
    dev_loader = DataLoader(dev_set, batch_size, valid_shuffle, collate_fn=collate_fn_kag)
    test_loader = DataLoader(test_set, batch_size, valid_shuffle, collate_fn=collate_fn_kag)

    return train_loader, dev_loader, test_loader
