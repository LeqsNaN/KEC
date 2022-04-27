# KEC
The code for the paper "Neutral Utterances are Also Causes: Enhancing Conversational Causal Emotion Entailment with Social Commonsense Knowledge". 

The appendix mentioned in the paper is present in [here](https://drive.google.com/file/d/1uuTwTjr8csn11BrLCqKx8IGkAbdjDQSp/view?usp=sharing). 

## Requirements
* Pytorch==1.8.1
* Transformers==4.3.3
* numpy=1.19.2
* nltk

## Additonal Data
Edge attributes of skaig: [skaig_data](https://drive.google.com/file/d/1oDCknwUuchL00byHhwhe_VjtHziJkQSn/view?usp=sharing)

## Training
```
bash run_single.sh
```

## Some explanations
`knowledge_select.py` is used to select sentimental related pieces of knowledge for a pair of utterances. 
`entail_construct.py` is used to form the data into the entailment style with or without emotion words. The generatad files is used to train and evaluate the [baseline of RECCON-DD](https://github.com/declare-lab/RECCON). 
