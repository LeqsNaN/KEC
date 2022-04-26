import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer


class BaseDataset(Dataset):
    def __init__(self, model_size='large', phase='train'):
        super(BaseDataset, self).__init__()
        self.emotion_dict = {'happiness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
        self.act_mapping = {'inform': 0, 'question': 1, 'directive': 2, 'commissive': 3}
        self.speaker_dict = {'A': 0, 'B': 1}
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-'+model_size)
        data_path = 'dd_data/dailydialog_' + phase + '.pkl'
        processed_data_path = 'dd_data/dailydialog_' + phase + '_processed.pkl'
        if os.path.exists(processed_data_path):
            data = pickle.load(open(processed_data_path, 'rb'), encoding='latin1')
            self.utterance = data[0]
            self.token_ids = data[1]
            self.attention_mask = data[2]
            self.emotion = data[3]
            self.adj_index = data[4]
            self.evidence = data[5]
            self.mask = data[6]
            self.conv_len = data[7]
            self.act_label = data[8]
        else:
            data = pickle.load(open(data_path, 'rb'), encoding='latin1')
            self.utterance = []
            self.token_ids = []
            self.attention_mask = []
            self.emotion = []
            self.adj_index = []
            self.evidence = []
            self.mask = []
            self.conv_len = []
            self.act_label = []
            for d in data:
                utter = []
                tk_id = []
                at_mk = []
                emo = []
                spk = []
                evi = []
                act_lbl = []
                self.conv_len.append(len(d))
                for utt in d:
                    u = utt['utterance']
                    encoded_output = self.tokenizer(u)
                    tid = encoded_output.input_ids
                    atm = encoded_output.attention_mask
                    tk_id.append(torch.tensor(tid, dtype=torch.long))
                    at_mk.append(torch.tensor(atm))
                    e = utt['emotion']
                    s = utt['speaker']
                    i = utt['id']
                    a = utt['act']
                    if 'evidence' in utt:
                        ev = utt['evidence']
                    else:
                        ev = []
                    utter.append(u)
                    emo.append(self.emotion_dict[e])
                    spk.append(self.speaker_dict[s])
                    ev_vic = [0] * len(d)
                    for i in ev:
                        if i != 'b':
                            ev_vic[i - 1] = 1
                    evi.append(ev_vic)
                    act_lbl.append(self.act_mapping[a])
                evi = torch.tensor(evi, dtype=torch.long)
                msk = torch.ones_like(evi, dtype=torch.long).tril(0)
                spk = torch.tensor(spk)
                act_lbl = torch.tensor(act_lbl)
                same_spk = spk.unsqueeze(1) == spk.unsqueeze(0)
                other_spk = same_spk.eq(False).long().tril(0)
                same_spk = same_spk.long().tril(0)
                # [2, conv_len, conv_len]
                spker = torch.stack([same_spk, other_spk], dim=0)
                tk_id = pad_sequence(tk_id, batch_first=True, padding_value=1)
                at_mk = pad_sequence(at_mk, batch_first=True, padding_value=0)
                emo = torch.tensor(emo, dtype=torch.long)
                self.utterance.append(utter)
                self.token_ids.append(tk_id)
                self.attention_mask.append(at_mk)
                self.emotion.append(emo)
                self.evidence.append(evi)
                self.mask.append(msk)
                self.adj_index.append(spker)
                self.act_label.append(act_lbl)
            to_be_saved_data = [self.utterance, self.token_ids, self.attention_mask, self.emotion,
                                self.adj_index, self.evidence, self.mask, self.conv_len, self.act_label]
            pickle.dump(to_be_saved_data, open(processed_data_path, 'wb'))

    def __getitem__(self, item):
        utter = self.utterance[item]
        # [conv_len, utter_len]
        token_id = self.token_ids[item]
        att_mask = self.attention_mask[item]
        label = self.emotion[item]
        # [conv_len, conv_len]
        adj_idx = self.adj_index[item]
        evid = self.evidence[item]
        msk = self.mask[item]
        clen = self.conv_len[item]
        act = self.act_label[item]
        # # [num_pair]
        # evi_pair = torch.masked_select(evid, msk.eq(1))
        return utter, token_id, att_mask, label, adj_idx, evid, msk, clen, act

    def __len__(self):
        return len(self.emotion)


class KnowledgeDataset(Dataset):
    def __init__(self, model_size='base', phase='train', window=10, csk_window=10):
        super(Dataset, self).__init__()
        self.base_dataset = BaseDataset(model_size, phase)
        self.window = window
        self.csk_window = csk_window
        knowledge_path = 'dd_data/dailydialog_' + phase + '_know_processed.pkl'
        processed_knowledge_path = 'dd_data/dailydialog_' + phase + '_know_tokenize.pkl'
        if os.path.exists(processed_knowledge_path):
            data = pickle.load(open(processed_knowledge_path, 'rb'), encoding='latin1')
            knowledge, attention_mask, knowledge_adj = data[0], data[1], data[2]
            self.knowledge = knowledge
            self.knowledge_attention_mask = attention_mask
            self.knowledge_adj = knowledge_adj
        else:
            knowledge, knowledge_adj = pickle.load(open(knowledge_path, 'rb'), encoding='latin1')
            knowledge_processed = []
            knowledge_attention_mask_processed = []
            knowledge_adj_processed = []
            for csk, kaj in zip(knowledge, knowledge_adj):
                csk = [c.strip('</s></s>') for c in csk]
                # csk = [' ' + c.strip(' ') for c in csk]
                csk = self.base_dataset.tokenizer(csk, padding=True, return_tensors='pt')
                knowledge_tensor = csk.input_ids
                knowledge_attention_mask = csk.attention_mask
                kaj = torch.tensor(kaj, dtype=torch.long)
                knowledge_processed.append(knowledge_tensor)
                knowledge_attention_mask_processed.append(knowledge_attention_mask)
                knowledge_adj_processed.append(kaj)
            self.knowledge = knowledge_processed
            self.knowledge_attention_mask = knowledge_attention_mask_processed
            self.knowledge_adj = knowledge_adj_processed
            save_data = [knowledge_processed, knowledge_attention_mask_processed, knowledge_adj_processed]
            pickle.dump(save_data, open(processed_knowledge_path, 'wb'))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, item):
        utter, token_id, att_mask, label, adj_idx, evid, msk, clen, act = self.base_dataset[item]
        edge_mask = msk.clone()
        knowledge = self.knowledge[item]
        knowledge_adj = self.knowledge_adj[item]
        knowledge_attention_mask = self.knowledge_attention_mask[item]
        knowledge_adj = knowledge_adj.triu(-self.csk_window)
        edge_mask = edge_mask.triu(-self.window)
        adj_idx = adj_idx.triu(-self.window)
        return utter, token_id, att_mask, label, adj_idx, evid, msk, clen, \
               act, knowledge, knowledge_attention_mask, knowledge_adj, edge_mask


def collate_fn_know(data):
    token_ids = []
    attention_mask = []
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
    knowledge_attention_mask = []
    knowledge_adj = []
    for i, d in enumerate(data):
        if i == 0:
            token_ids = d[1]
            attention_mask = d[2]
        else:
            d_tids = d[1]
            max_len = max(token_ids.shape[1], d_tids.shape[1])
            if token_ids.shape[1] < max_len:
                token_ids = torch.cat([token_ids, torch.ones(token_ids.shape[0], max_len - token_ids.shape[1], dtype=torch.long)], dim=1)
            if d_tids.shape[1] < max_len:
                d_tids = torch.cat([d_tids, torch.ones(d_tids.shape[0], max_len - d_tids.shape[1], dtype=torch.long)], dim=1)
            token_ids = torch.cat([token_ids, d_tids], dim=0)

            a_msk = d[2]
            max_len = max(attention_mask.shape[1], a_msk.shape[1])
            if attention_mask.shape[1] < max_len:
                attention_mask = torch.cat([attention_mask, torch.zeros(attention_mask.shape[0], max_len - attention_mask.shape[1], dtype=torch.long)], dim=1)
            if a_msk.shape[1] < max_len:
                a_msk = torch.cat([a_msk, torch.zeros(a_msk.shape[0], max_len - a_msk.shape[1], dtype=torch.long)], dim=1)
            attention_mask = torch.cat([attention_mask, a_msk], dim=0)
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
        knowledge_attention_mask.append(d[10])
        knowledge_adj.append(d[11])
        edge_mask.append(d[12])
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
    max_knowledge_len = max([k.shape[1] for k in knowledge])
    num_knowledge = [k.shape[0] for k in knowledge]
    knowledge = [torch.cat([k, torch.ones(k.shape[0], max_knowledge_len-k.shape[1])], dim=1) for k in knowledge]
    knowledge = torch.cat(knowledge, dim=0)
    knowledge_attention_mask = [torch.cat([km, torch.zeros(km.shape[0], max_knowledge_len-km.shape[1])], dim=1) for km in knowledge_attention_mask]
    knowledge_attention_mask = torch.cat(knowledge_attention_mask, dim=0)
    accu_know = 0
    for i in range(len(num_knowledge)):
        knowledge_adj[i] = knowledge_adj[i] + accu_know
        accu_know += num_knowledge[i]
        knowledge_adj[i] = torch.cat([torch.cat([knowledge_adj[i], torch.zeros(max_len - knowledge_adj[i].shape[0], knowledge_adj[i].shape[1])], dim=0), torch.zeros(max_len, max_len-knowledge_adj[i].shape[1])], dim=1)
    knowledge_adj = torch.stack(knowledge_adj, dim=0)

    return token_ids, attention_mask, token_ids_1, attention_mask_1, clen, mask, edge_mask, adj_index, \
           label, ece_pair, act_label, knowledge, knowledge_attention_mask, knowledge_adj, num_knowledge


def collate_fn(data):
    token_ids = []
    attention_mask = []
    label = []
    adj_index = []
    ece_pair = []
    mask = []
    clen = []
    act_label = []
    for i, d in enumerate(data):
        if i == 0:
            token_ids = d[1]
            attention_mask = d[2]
        else:
            d_tids = d[1]
            max_len = max(token_ids.shape[1], d_tids.shape[1])
            if token_ids.shape[1] < max_len:
                token_ids = torch.cat([token_ids, torch.ones(token_ids.shape[0], max_len-token_ids.shape[1], dtype=torch.long)], dim=1)
            if d_tids.shape[1] < max_len:
                d_tids = torch.cat([d_tids, torch.ones(d_tids.shape[0], max_len - d_tids.shape[1], dtype=torch.long)], dim=1)
            token_ids = torch.cat([token_ids, d_tids], dim=0)

            a_msk = d[2]
            max_len = max(attention_mask.shape[1], a_msk.shape[1])
            if attention_mask.shape[1] < max_len:
                attention_mask = torch.cat([attention_mask, torch.zeros(attention_mask.shape[0], max_len-attention_mask.shape[1], dtype=torch.long)], dim=1)
            if a_msk.shape[1] < max_len:
                a_msk = torch.cat([a_msk, torch.zeros(a_msk.shape[0], max_len-a_msk.shape[1], dtype=torch.long)], dim=1)
            attention_mask = torch.cat([attention_mask, a_msk], dim=0)
        label.append(d[3])
        # [2, conv_len, conv_len]
        adj_index.append(d[4])
        ece_pair.append(d[5])
        mask.append(d[6])
        clen.append(d[7])
        act_label.append(d[8])
    label = pad_sequence(label, batch_first=True, padding_value=-1)
    act_label = pad_sequence(act_label, batch_first=True, padding_value=-1)
    max_len = max(clen)
    mask = [torch.cat([torch.cat([m, torch.zeros(max_len-m.shape[0], m.shape[1])], dim=0), torch.zeros(max_len, max_len-m.shape[1])], dim=1) for m in mask]
    mask = torch.stack(mask, dim=0)
    ece_pair = [torch.cat([torch.cat([ep, torch.zeros(max_len-ep.shape[0], ep.shape[1])], dim=0), torch.zeros(max_len, max_len-ep.shape[1])], dim=1) for ep in ece_pair]
    ece_pair = torch.stack(ece_pair, dim=0)
    adj_index = [torch.cat([torch.cat([a, torch.zeros(2, max_len-a.shape[1], a.shape[2])], dim=1), torch.zeros(2, max_len, max_len-a.shape[2])], dim=2) for a in adj_index]
    # [batch_size, 2, conv_len, conv_len]
    adj_index = torch.stack(adj_index, dim=0)

    return token_ids, attention_mask, clen, mask, adj_index, label, ece_pair, act_label


def get_dataloaders(model_size, batch_size, valid_shuffle, dataset_type='base', window=10, csk_window=10):
    if dataset_type == 'base':
        train_set = BaseDataset(model_size, 'train')
        dev_set = BaseDataset(model_size, 'dev')
        test_set = BaseDataset(model_size, 'test')
        train_loader = DataLoader(train_set, batch_size, True, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_set, batch_size, valid_shuffle, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size, valid_shuffle, collate_fn=collate_fn)
    else:
        train_set = KnowledgeDataset(model_size, 'train', window, csk_window)
        dev_set = KnowledgeDataset(model_size, 'dev', window, csk_window)
        test_set = KnowledgeDataset(model_size, 'test', window, csk_window)
        train_loader = DataLoader(train_set, batch_size, True, collate_fn=collate_fn_know)
        dev_loader = DataLoader(dev_set, batch_size, valid_shuffle, collate_fn=collate_fn_know)
        test_loader = DataLoader(test_set, batch_size, valid_shuffle, collate_fn=collate_fn_know)

    return train_loader, dev_loader, test_loader
