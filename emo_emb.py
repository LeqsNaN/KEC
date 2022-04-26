from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import pickle


emo = {'happiness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
emo_list = [' happiness', ' neutral', ' anger', ' sadness', ' fear', ' surprise', ' disgust']
# act_list = [' inform', ' question', ' directive', ' promise']

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

emo_list_ = [tokenizer.encode(e, add_special_tokens=False) for e in emo_list]
# act_list_ = [tokenizer.encode(e, add_special_tokens=False) for e in act_list]

vocab_weight = model.lm_head.decoder.weight

emo_emb = [vocab_weight[e] for e in emo_list_]
# act_emb = [vocab_weight[e] for e in act_list_]
print(emo_emb[0].shape)
# print(act_emb[0].shape)
emo_emb = [torch.zeros(1, vocab_weight.shape[1])] + emo_emb
# act_emb = [torch.zeros(1, vocab_weight.shape[1])] + act_emb
emo_emb = torch.cat(emo_emb, dim=0)
# act_emb = torch.cat(act_emb, dim=0)
pickle.dump(emo_emb, open('emotion_embeddings.pkl', 'wb'))
# pickle.dump(act_emb, open('act_embeddings.pkl', 'wb'))
