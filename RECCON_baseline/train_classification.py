import numpy as np, pandas as pd
import json, os, logging, pickle, argparse
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel
import pickle
import pandas as pd
import torch


def dataframe_trans(data_list):
    text = data_list[0]
    labels = data_list[1]
    new_data = []
    for tt, ll in zip(text, labels):
        new_data.append([tt, ll])
    df = pd.DataFrame(new_data, columns=['text', 'labels'])
    return df


if __name__ == '__main__':

    global args
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate') 
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=12, metavar='E', help='number of epochs')
    parser.add_argument('--model', default='rob', help='which model rob| robl')
    parser.add_argument('--fold', type=int, default=1, metavar='F', help='which fold')
    parser.add_argument('--emotion', action='store_true', default=False, help='use emotion')
    parser.add_argument('--cuda', type=int, default=0, metavar='C', help='cuda device')
    parser.add_argument('--seed', type=int, default=100, help='manual seed')
    args = parser.parse_args()

    print(args)
    
    model_family = {'rob': 'roberta', 'robl': 'roberta'}
    model_id = {'rob': 'roberta-base', 'robl': 'roberta-large'}
    model_exact_id = {'rob': 'roberta-base', 'robl': 'roberta-large'}
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    model = args.model
    fold = str(args.fold)
    emotion = args.emotion
    cuda = args.cuda
    dataset = 'dailydialog'
    seed = str(args.seed)
    manual_seed = args.seed
    
    max_seq_length = 512
        
    if emotion == False:
        save_dir    = 'outputs/' + model_id[model] + '-dailydialog-cls-without-emotion-fold' + fold + '-seed-' + seed + '/'
        result_file = 'outputs/' + model_id[model] + '-dailydialog-cls-without-emotion-fold' + fold + '-seed-' + seed + '/results.txt'
        dump_file   = 'outputs/' + model_id[model] + '-dailydialog-cls-without-emotion-fold' + fold + '-seed-' + seed + '/test_predictions.pkl'
        x_train = pickle.load(open('data/subtask2/fold' + fold + '/dailydialog_classification_train_without_emotion.pkl', 'rb'), encoding='latin1')
        x_valid = pickle.load(open('data/subtask2/fold' + fold + '/dailydialog_classification_dev_without_emotion.pkl', 'rb'), encoding='latin1')
        x_test  = pickle.load(open('data/subtask2/fold' + fold + '/dailydialog_classification_test_without_emotion.pkl', 'rb'), encoding='latin1')
    else:
        save_dir    = 'outputs/' + model_id[model] + '-dailydialog-cls-with-emotion-fold' + fold + '-seed-' + seed + '/'
        result_file = 'outputs/' + model_id[model] + '-dailydialog-cls-with-emotion-fold' + fold + '-seed-' + seed + '/results.txt'
        dump_file   = 'outputs/' + model_id[model] + '-dailydialog-cls-with-emotion-fold' + fold + '-seed-' + seed + '/test_predictions.pkl'
        x_train = pickle.load(open('data/subtask2/fold' + fold + '/dailydialog_classification_train_with_emotion.pkl', 'rb'), encoding='latin1')
        x_valid = pickle.load(open('data/subtask2/fold' + fold + '/dailydialog_classification_dev_with_emotion.pkl', 'rb'), encoding='latin1')
        x_test  = pickle.load(open('data/subtask2/fold' + fold + '/dailydialog_classification_test_with_emotion.pkl', 'rb'), encoding='latin1')
    
    x_train = dataframe_trans(x_train)
    x_valid = dataframe_trans(x_valid)
    x_test = dataframe_trans(x_test)
    
    if fold == '1':
        num_steps = int(27915/batch_size)
    else:
        num_steps = int(25697/batch_size)
    
    train_args = {
        'fp16': False,
        'overwrite_output_dir': True, 
        'max_seq_length': max_seq_length,
        'learning_rate': lr,
        'sliding_window': False,
        'output_dir': save_dir,
        'best_model_dir': save_dir + 'best_model/',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': num_steps,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': 500000,
        'train_batch_size': batch_size,
        'num_train_epochs': epochs, 
        'manual_seed': manual_seed
    }
    
    cls_model = ClassificationModel(model_family[model], model_exact_id[model], args=train_args, cuda_device=cuda)
    cls_model.train_model(x_train, eval_df=x_valid)
    
    cls_model = ClassificationModel(model_family[model], save_dir + 'best_model/', args=train_args, cuda_device=cuda)
    result, model_outputs, wrong_predictions = cls_model.eval_model(x_test)
    
    preds = np.argmax(model_outputs, 1)
    labels = x_test['labels']

    r = str(classification_report(labels, preds, digits=4))
    print (r)
    
    rf = open('results/dailydialog_classification.txt', 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*54 + '\n')    
    rf.close()
    
    rf = open(result_file, 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*54 + '\n')    
    rf.close()
    
    pickle.dump([result, model_outputs, wrong_predictions], open(dump_file, 'wb'))
    
