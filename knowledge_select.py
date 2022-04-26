import torch
import numpy as np
import pickle
import nltk
from nltk import word_tokenize
from nltk.corpus import sentiwordnet as swn


emotion_dict = {'happiness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
emotion_mapping = {0: 1, 1: 0, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1}
senti_mapping = {-1: '-', 0: 'o', 1: '+'}
know_edge = {'o+': 1, 'o-': 2, '+': 3, '-': 4}


def text_score(text):
    if text == '':
        return []
    if text == ' none':
        return []
    tag_seq = nltk.pos_tag([i for i in word_tokenize(str(text).lower())])

    key = []
    pos = []

    n = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']
    v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    a = ['JJ', 'JJR', 'JJS']
    r = ['RB', 'RBR', 'RBS', 'RP', 'WRB']

    for i in range(len(tag_seq)):
        key.append(tag_seq[i][0])
        if tag_seq[i][1] in n:
            pos.append('n')
        elif tag_seq[i][1] in v:
            pos.append('v')
        elif tag_seq[i][1] in a:
            pos.append('a')
        elif tag_seq[i][1] in r:
            pos.append('r')
        else:
            pos.append('')

    pos_score = 0.
    neg_score = 0.
    obj_score = 0.
    num_words = 0
    for i in range(len(key)):
        m = list(swn.senti_synsets(key[i], pos[i]))
        if len(m) > 0:
            num_words += 1
            selected_synset = m[0]
            pos_score = pos_score + selected_synset.pos_score()
            neg_score = neg_score + selected_synset.neg_score()
            obj_score = obj_score + selected_synset.obj_score()
    if num_words != 0:
        pos_score = pos_score / num_words
        neg_score = neg_score / num_words
        obj_score = obj_score / num_words
        sen_score = pos_score - neg_score
        sen_score = sen_score if abs(sen_score) > obj_score else 0.
        if sen_score != 0.:
            sen_score = 1 if sen_score > 0 else -1
        return [text, sen_score]
    else:
        return []


def read_data(data_path, knowledge_path, processed_knowledge_path, window=10):
    data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    emotion_label = data[3]
    # evident_label = data[5]
    knowledge = pickle.load(open(knowledge_path, 'rb'), encoding='latin1')
    all_conversation_selected_knowledge = []
    all_conversation_selected_edge_adj = []
    for conv_id in range(len(emotion_label)):
        conv_know = [' none</s></s>']
        # conv_know = [' none ']
        conversation_emotion = emotion_label[conv_id].numpy().tolist()
        # conversation_evidence = evident_label[conv_id].numpy().tolist()
        conversation_know = knowledge[conv_id]
        conv_len = len(conversation_emotion)
        edge_type_adj = np.zeros((conv_len, conv_len), dtype=np.int)
        edge_index_adj = np.zeros((conv_len, conv_len), dtype=np.int)
        edge_index = 1
        for i in range(len(conversation_emotion)):
            source_emotion = conversation_emotion[i]
            source_sentiment = emotion_mapping[source_emotion]
            source_senti_token = senti_mapping[source_sentiment]
            source_knowledge = conversation_know[i]
            f_item = min(conv_len, i+window+1)
            source_xEffect = source_knowledge['xEffect'].split(' ==sep== ')
            source_xEffect = [text_score(k) for k in source_xEffect]
            source_oEffect = source_knowledge['oEffect'].split(' ==sep== ')
            source_oEffect = [text_score(k) for k in source_oEffect]
            source_xReact = source_knowledge['xReact'].split(' ==sep== ')
            source_xReact = [text_score(k) for k in source_xReact]
            source_oReact = source_knowledge['oReact'].split(' ==sep== ')
            source_oReact = [text_score(k) for k in source_oReact]
            source_same_speaker_know = {'o': ' none</s></s>', '+': ' none</s></s>', '-': ' none</s></s>'}
            # source_same_speaker_know = {'o': ' none ', '+': ' none ', '-': ' none '}
            neu_know = ''
            pos_know = ''
            neg_know = ''
            for item in source_xEffect:
                if len(item) == 0:
                    continue
                kn, sent = item
                if sent == 0:
                    neu_know = neu_know + kn + '</s></s>'
                    # neu_know = neu_know + kn
                elif sent == 1:
                    pos_know = pos_know + kn + '</s></s>'
                    # pos_know = pos_know + kn
                elif sent == -1:
                    neg_know = neg_know + kn + '</s></s>'
                    # neg_know = neg_know + kn
                else:
                    raise NotImplementedError()
            for item in source_xReact:
                if len(item) == 0:
                    continue
                kn, sent = item
                if sent == 0:
                    neu_know = neu_know + kn + '</s></s>'
                    # neu_know = neu_know + kn
                elif sent == 1:
                    pos_know = pos_know + kn + '</s></s>'
                    # pos_know = pos_know + kn
                elif sent == -1:
                    neg_know = neg_know + kn + '</s></s>'
                    # neg_know = neg_know + kn
                else:
                    raise NotImplementedError()
            if neu_know != '':
                source_same_speaker_know['o'] = neu_know
            if pos_know != '':
                source_same_speaker_know['+'] = pos_know
            if neg_know != '':
                source_same_speaker_know['-'] = neg_know
            source_different_speaker_know = {'o': ' none</s></s>', '+': ' none</s></s>', '-': ' none</s></s>'}
            # source_different_speaker_know = {'o': ' none ', '+': ' none ', '-': ' none '}
            neu_know = ''
            pos_know = ''
            neg_know = ''
            for item in source_oEffect:
                if len(item) == 0:
                    continue
                kn, sent = item
                if sent == 0:
                    neu_know = neu_know + kn + '</s></s>'
                    # neu_know = neu_know + kn
                elif sent == 1:
                    pos_know = pos_know + kn + '</s></s>'
                    # pos_know = pos_know + kn
                elif sent == -1:
                    neg_know = neg_know + kn + '</s></s>'
                    # neg_know = neg_know + kn
                else:
                    raise NotImplementedError()
            for item in source_oReact:
                if len(item) == 0:
                    continue
                kn, sent = item
                if sent == 0:
                    neu_know = neu_know + kn + '</s></s>'
                    # neu_know = neu_know + kn
                elif sent == 1:
                    pos_know = pos_know + kn + '</s></s>'
                    # pos_know = pos_know + kn
                elif sent == -1:
                    neg_know = neg_know + kn + '</s></s>'
                    # neg_know = neg_know + kn
                else:
                    raise NotImplementedError()
            if neu_know != '':
                source_different_speaker_know['o'] = neu_know
            if pos_know != '':
                source_different_speaker_know['+'] = pos_know
            if neg_know != '':
                source_different_speaker_know['-'] = neg_know
            same_speaker = True
            edge_buffer = {}
            for j in range(i, f_item):
                target_emotion = conversation_emotion[j]
                target_sentiment = emotion_mapping[target_emotion]
                target_senti_token = senti_mapping[target_sentiment]
                if source_senti_token == 'o':
                    interaction = source_senti_token + target_senti_token
                else:
                    interaction = target_senti_token
                if interaction in know_edge:
                    edge_type_adj[j, i] = know_edge[interaction]
                    if same_speaker:
                        interaction_speaker = interaction + 's'
                        if interaction_speaker in edge_buffer:
                            edge_index_adj[j, i] = edge_buffer[interaction_speaker]
                        else:
                            if source_senti_token == 'o':
                                klg = source_same_speaker_know['o'] + source_same_speaker_know[target_senti_token]
                            else:
                                klg = source_same_speaker_know[target_senti_token]
                            klg = source_same_speaker_know[target_senti_token]
                            if klg.strip(' ') != conv_know[0].strip(' '):
                                conv_know.append(klg)
                                edge_buffer[interaction_speaker] = edge_index
                                edge_index += 1
                            else:
                                edge_buffer[interaction_speaker] = 0
                            edge_index_adj[j, i] = edge_buffer[interaction_speaker]
                        same_speaker = False
                    else:
                        interaction_speaker = interaction + 'd'
                        if interaction_speaker in edge_buffer:
                            edge_index_adj[j, i] = edge_buffer[interaction_speaker]
                        else:
                            if source_senti_token == 'o':
                                klg = source_different_speaker_know['o'] + source_different_speaker_know[target_senti_token]
                            else:
                                klg = source_different_speaker_know[target_senti_token]
                            klg = source_different_speaker_know[target_senti_token]
                            if klg != conv_know[0]:
                                conv_know.append(klg)
                                edge_buffer[interaction_speaker] = edge_index
                                edge_index += 1
                            else:
                                edge_buffer[interaction_speaker] = 0
                            edge_index_adj[j, i] = edge_buffer[interaction_speaker]
                        same_speaker = True
                else:
                    edge_type_adj[j, i] = 0
                    edge_index_adj[j, i] = 0
        all_conversation_selected_knowledge.append(conv_know)
        all_conversation_selected_edge_adj.append(edge_index_adj)
    print(all_conversation_selected_knowledge[0])
    print(all_conversation_selected_edge_adj[0])
    pickle.dump([all_conversation_selected_knowledge, all_conversation_selected_edge_adj], open(processed_knowledge_path, 'wb'))


if __name__ == '__main__':
    read_data('dd_data/dailydialog_train_processed.pkl',
              'dd_data/dailydialog_train_know.pkl',
              'dd_data/dailydialog_train_know_processed.pkl')
    read_data('dd_data/dailydialog_dev_processed.pkl',
              'dd_data/dailydialog_dev_know.pkl',
              'dd_data/dailydialog_dev_know_processed.pkl')
    read_data('dd_data/dailydialog_test_processed.pkl',
              'dd_data/dailydialog_test_know.pkl',
              'dd_data/dailydialog_test_know_processed.pkl')
