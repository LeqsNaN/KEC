import json
import torch
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]


def get_knowledge(model, dataset):
    knowledge_set = []
    relation_set = ["oEffect", "oReact", "oWant", "xAttr", "xEffect",
                    "xIntent", "xNeed", "xReact", "xReason", "xWant"]
    # relation_set = ["oReact", "xReact"]
    count = 1
    for conv in dataset:
        conv_knowledge = []
        for utter in conv:
            print(f'{count} processed')
            utter_knowledge = {}
            # utter_knowledge = 'this person feels'
            queries = []
            utterance = utter['utterance']
            for r in relation_set:
                query = "{} {} [GEN]".format(utterance, r)
                queries.append(query)
            results = model.generate(queries, decode_method="beam", num_generate=5)
            for relation, result in zip(relation_set, results):
                utter_knowledge[relation] = ' ==sep== '.join(result)
                # if relation == 'oReact':
                #     utter_knowledge = utter_knowledge + result[0] + ' and' + result[1] + ' . '
                # else:
                #     utter_knowledge = utter_knowledge + 'others feels' + result[0] + ' and ' + result[1] + ' . '
            conv_knowledge.append(utter_knowledge)
            count += 1
        knowledge_set.append(conv_knowledge)
    return knowledge_set


if __name__ == "__main__":

    # sample usage
    print("model loading ...")
    comet = Comet("./comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")

    train_data = pickle.load(open('/data2/ljn/RECCON-ERC/dd_data/dailydialog_train.pkl', 'rb'), encoding='latin1')
    dev_data = pickle.load(open('/data2/ljn/RECCON-ERC/dd_data/dailydialog_dev.pkl', 'rb'), encoding='latin1')
    test_data = pickle.load(open('/data2/ljn/RECCON-ERC/dd_data/dailydialog_test.pkl', 'rb'), encoding='latin1')

    print('train data')
    train_knowledge = get_knowledge(comet, train_data)
    print('dev data')
    dev_knowledge = get_knowledge(comet, dev_data)
    print('test data')
    test_knowledge = get_knowledge(comet, test_data)

    pickle.dump(train_knowledge, open('/data2/ljn/RECCON-ERC/dd_data/dailydialog_train_know.pkl', 'wb'))
    pickle.dump(dev_knowledge, open('/data2/ljn/RECCON-ERC/dd_data/dailydialog_dev_know.pkl', 'wb'))
    pickle.dump(test_knowledge, open('/data2/ljn/RECCON-ERC/dd_data/dailydialog_test_know.pkl', 'wb'))
