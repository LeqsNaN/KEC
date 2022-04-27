The code for the paper "Neutral Utterances are Also Causes: Enhancing Conversational Causal Emotion Entailment with Social Commonsense Knowledge". 

The appendix mentioned in the paper is present in [here](https://drive.google.com/file/d/1uuTwTjr8csn11BrLCqKx8IGkAbdjDQSp/view?usp=sharing). 

Some code is based on [DAG-ERC](https://github.com/shenwzh3/DAG-ERC), [RECCON](https://github.com/declare-lab/RECCON), and [COMET-ATOMIC-2020](https://github.com/allenai/comet-atomic-2020). 

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
* `generate_knowledge.py` is used to generate social commonsense knowledge for our method. Put it in `comet-atomic-2020/models/comet_atomic2020_bart/` of [COMET-ATOMIC-2020](https://github.com/allenai/comet-atomic-2020). P.S. the paths of loaded and dumped files should be modified to your own data paths. We have uploaded all the generated knowledge data in `dd_data`. 

* `knowledge_select.py` is used to select sentimental related pieces of knowledge for a pair of utterances. We have uploaded all the processed data in `dd_data`. 

* `entail_construct.py` is used to form the data into the entailment style with or without emotion words. The generatad files is used to train and evaluate the [baseline of RECCON-DD](https://github.com/declare-lab/RECCON). Furthermore, replace the `train_classification.py`, `eval_classification.py` in [RECCON](https://github.com/declare-lab/RECCON)  with `RECCON_baseline/train_classification.py` and `RECCON_baseline/eval_classification.py` in this repository. We have uploaded the entailment style data in [here](https://drive.google.com/file/d/1qlwtdvkwjCKDpdG1uUVgrckS8erV1w4y/view?usp=sharing). Download the data and put them in `data/subtask2/fold1/` in [RECCON](https://github.com/declare-lab/RECCON). 
