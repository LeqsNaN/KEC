import pickle


def get_data(phase):
    data_path = 'dd_data/dailydialog_' + phase + '.pkl'
    data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    utterance_we = []
    utterance_woe = []
    target_emotion = []
    source_emotion = []
    label_list = []
    for conv in data:
        conv_len = len(conv)
        conv_history = ''
        for i in range(conv_len):
            target_utt = conv[i]['utterance']
            if i == 0:
                conv_history = target_utt
            else:
                conv_history = ' '.join([conv_history, target_utt])
            target_emo = conv[i]['emotion']
            if 'evidence' in conv[i]:
                target_evi = conv[i]['evidence']
            else:
                target_evi = []
            for j in range(i+1):
                source_utt = conv[j]['utterance']
                source_emo = conv[j]['emotion']
                target_emotion.append(target_emo)
                source_emotion.append(source_emo)
                if j+1 in target_evi:
                    label = 1
                else:
                    label = 0
                data_item_with_emotion = ' <SEP> '.join([target_emo, target_utt, source_utt, conv_history])
                data_item_without_emotion = ' <SEP> '.join([target_utt, source_utt, conv_history])
                utterance_we.append(data_item_with_emotion)
                utterance_woe.append(data_item_without_emotion)
                label_list.append(label)
    data_we = [utterance_we, label_list, target_emotion, source_emotion]
    data_woe = [utterance_woe, label_list, target_emotion, source_emotion]

    pickle.dump(data_we, open('dailydialog_' + phase + '_with_emotion.pkl', 'wb'))
    pickle.dump(data_woe, open('dailydialog_' + phase + '_without_emotion.pkl', 'wb'))


# get_data('train')
# get_data('dev')
get_data('test')
