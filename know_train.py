import random
import numpy as np
import os
import torch
import pickle
import argparse
from dag import CauseDag
from dag1 import CauseDagNoEmotion
from know_model_kec import CskCauseDag
from know_model_0_window import CskCauseDag0
from ck_transformer import CskTransformer
from utils import MaskedBCELoss2
from dataset import get_dataloaders
from sklearn.metrics import f1_score, classification_report
from transformers import AdamW, get_constant_schedule, get_linear_schedule_with_warmup


def train(model_variant, model, model_path, train_loader, dev_loader, test_loader,
          loss_fn, optimizer, n_epochs, log, accumulate_step, scheduler):
    model.train()
    best_f1_score_ece = 0.
    best_f1_score_ece_test = 0.

    best_report_ece = None
    best_report_ece_test = None

    step = 0

    for e in range(n_epochs):
        ece_prediction_list = []
        ece_prediction_mask = []
        ece_label_list = []
        ece_loss = 0.
        ece_total_sample = 0

        for data in train_loader:
            input_ids, attention_mask, input_ids_1, attention_mask_1, clen, mask, edge_mask, adj_index, \
                label, ece_pair, _, knowledge_text, knowledge_mask, know_adj, num_know = data
            # input_ids = input_ids.cuda()
            # attention_mask = attention_mask.cuda()
            mask = mask.cuda()
            label = label.cuda()
            ece_pair = ece_pair.cuda()
            knowledge_text = knowledge_text.long().cuda()
            knowledge_mask = knowledge_mask.cuda()
            know_adj = know_adj.long().cuda()
            edge_mask = edge_mask.float().cuda()
            s_mask = adj_index[:, 0, :, :].float().cuda()
            o_mask = adj_index[:, 1, :, :].float().cuda()
            input_ids_1 = [t.cuda() for t in input_ids_1]
            attention_mask_1 = [t.cuda() for t in attention_mask_1]
            if model_variant == 'no_csk':
                # prediction = model(input_ids, attention_mask, clen, mask.float(), s_mask, o_mask, edge_mask, label+1)
                prediction = model(input_ids_1, attention_mask_1, clen, mask.float(), s_mask, o_mask, edge_mask, label+1)
            else:
                if model_variant == 'csk':
                    prediction = model(input_ids_1, attention_mask_1, clen, mask.float(), s_mask, o_mask,
                                       edge_mask, label+1, knowledge_text, knowledge_mask, know_adj)
                else:
                    prediction = model(input_ids_1, attention_mask_1, clen, edge_mask, s_mask, o_mask,
                                       mask.float(), label+1, knowledge_text, knowledge_mask, know_adj, False)
            ece_label_list.append(torch.flatten(ece_pair.data).cpu().numpy())
            ece_prediction_mask.append(torch.flatten(mask.data).cpu().numpy())
            ece_samples = mask.data.sum().item()
            ece_total_sample += ece_samples
            loss = loss_fn(ece_pair, prediction, mask)
            ece_loss = ece_loss + loss.item() * ece_samples
            ece_prediction = torch.gt(prediction.data, 0.5).long()
            ece_prediction_list.append(torch.flatten(ece_prediction.data).cpu().numpy())

            if accumulate_step > 1:
                loss = loss / accumulate_step
            model.zero_grad()
            loss.backward()
            if (step + 1) % accumulate_step == 0:
                optimizer.step()
                scheduler.step()
            step += 1
        ece_prediction_mask = np.concatenate(ece_prediction_mask)
        ece_label_list = np.concatenate(ece_label_list)

        ece_loss = ece_loss / ece_total_sample
        ece_prediction = np.concatenate(ece_prediction_list)
        fscore_ece = f1_score(ece_label_list, ece_prediction,
                              average='macro', sample_weight=ece_prediction_mask)
        log_line = f'[Train] Epoch {e+1}: ECE loss: {round(ece_loss, 6)}, Fscore: {round(fscore_ece, 4)}'
        print(log_line)
        log.write(log_line + '\n')

        dev_fscores, dev_reports = valid('DEV', model_variant, model, dev_loader, log)
        test_fscores, test_reports = valid('TEST', model_variant, model, test_loader, log)

        ece_final_fscore_dev = dev_fscores[0]
        ece_final_fscore_test = test_fscores[0]
        if best_f1_score_ece < ece_final_fscore_dev:
            best_f1_score_ece = ece_final_fscore_dev
            best_f1_score_ece_test = ece_final_fscore_test
            best_report_ece = dev_reports
            best_report_ece_test = test_reports
            torch.save(model, model_path)

    log_line = f'[FINAL--DEV]: best_Fscore: {round(best_f1_score_ece, 4)}'
    print(log_line)
    log.write('\n\n' + log_line + '\n\n')
    log.write(best_report_ece + '\n')

    log_line = f'[FINAL--TEST]: best_Fscore: {round(best_f1_score_ece_test, 4)}'
    print(log_line)
    log.write('\n\n' + log_line + '\n\n')
    log.write(best_report_ece_test + '\n')
    log.close()

    return best_f1_score_ece, best_f1_score_ece_test


def valid(valid_type, model_variant, model, data_loader, log):
    model.eval()
    ece_prediction_list = []
    ece_prediction_mask = []
    ece_label_list = []
    ece_total_sample = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids, attention_mask, input_ids_1, attention_mask_1, clen, mask, edge_mask, adj_index, \
                label, ece_pair, _, knowledge_text, know_mask, know_adj, num_know = data
            # input_ids = input_ids.cuda()
            # attention_mask = attention_mask.cuda()
            input_ids_1 = [t.cuda() for t in input_ids_1]
            attention_mask_1 = [t.cuda() for t in attention_mask_1]
            mask = mask.cuda()
            label = label.cuda()
            ece_pair = ece_pair.cuda()
            knowledge_text = knowledge_text.long().cuda()
            know_adj = know_adj.long().cuda()
            know_mask = know_mask.cuda()
            edge_mask = edge_mask.float().cuda()
            s_mask = adj_index[:, 0, :, :].float().cuda()
            o_mask = adj_index[:, 1, :, :].float().cuda()
            if model_variant == 'no_csk':
                # prediction = model(input_ids, attention_mask, clen, mask.float(), s_mask, o_mask, edge_mask, label+1)
                prediction = model(input_ids_1, attention_mask_1, clen, mask.float(), s_mask, o_mask, edge_mask, label+1)
            else:
                if model_variant == 'csk':
                    prediction = model(input_ids_1, attention_mask_1, clen, mask.float(), s_mask, o_mask,
                                       edge_mask, label+1, knowledge_text, know_mask, know_adj)
                else:
                    prediction = model(input_ids_1, attention_mask_1, clen, edge_mask, s_mask, o_mask,
                                       mask.float(), label+1, knowledge_text, know_mask, know_adj, False)
            ece_prediction = torch.flatten(torch.gt(prediction.data, 0.5).long()).cpu().numpy()
            ece_prediction_list.append(ece_prediction)

            ece_label_list.append(torch.flatten(ece_pair.data).cpu().numpy())
            ece_prediction_mask.append(torch.flatten(mask.data).cpu().numpy())
            ece_samples = mask.data.sum().item()
            ece_total_sample += ece_samples
    ece_prediction_mask = np.concatenate(ece_prediction_mask)
    ece_label_list = np.concatenate(ece_label_list)
    ece_prediction = np.concatenate(ece_prediction_list)
    fscore_ece = f1_score(ece_label_list,
                          ece_prediction,
                          average='macro',
                          sample_weight=ece_prediction_mask)
    log_line = f'[{valid_type}]: Fscore: {round(fscore_ece, 4)}'
    reports = classification_report(ece_label_list,
                                    ece_prediction,
                                    target_names=['neg', 'pos'],
                                    sample_weight=ece_prediction_mask,
                                    digits=4)
    print(log_line)
    log.write(log_line + '\n')
    fscores = [fscore_ece]
    model.train()

    return fscores, reports


def main(args, seed=0, index=0):
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_variant = args.model_variant
    model_path = args.save_dir
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = os.path.join(model_path, str(model_variant))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    batch_size = args.batch_size
    lr = args.lr
    model_name = 'model_' + str(index) + '.pkl'
    model_name = os.path.join(model_path, model_name)
    log_name = 'log_' + str(index) + '.txt'
    log_path = os.path.join(model_path, log_name)

    log = open(log_path, 'w')
    log.write(str(args) + '\n\n')

    model_size = args.model_size
    valid_shuffle = args.valid_shuffle
    window = args.window
    csk_window = args.csk_window

    # train_loader, dev_loader, test_loader = get_dataloaders(model_size, batch_size, valid_shuffle, window, csk_window)
    train_loader, dev_loader, test_loader = get_dataloaders(model_size, batch_size, valid_shuffle, 'know', window, csk_window)
    n_epochs = args.n_epochs
    mapping_type = args.mapping_type
    weight_decay = args.weight_decay
    utter_dim = args.utter_dim

    conv_encoder = args.conv_encoder
    rnn_dropout = args.rnn_dropout
    accumulate_step = args.accumulate_step
    scheduler_type = args.scheduler
    num_layers = args.num_layers

    emotion_dim = args.emotion_dim
    add_emotion = args.add_emotion
    dag_dropout = args.dag_dropout
    pooler_type = args.pooler_type

    nhead = args.nhead
    ff_dim = args.ff_dim
    att_dropout = args.att_dropout
    trm_dropout = args.trm_dropout
    max_len = args.max_len
    pe_type = args.pe_type

    multi_gpu = args.multigpu

    if add_emotion:
        emo_emb = pickle.load(open('emotion_embeddings.pkl', 'rb'), encoding='latin1')
    else:
        emo_emb = None

    if model_variant == 'no_csk':
        if add_emotion:
            model = CauseDag(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout,
                             num_layers, dag_dropout, pooler_type, emo_emb, emotion_dim)
        else:
            model = CauseDagNoEmotion(model_size, mapping_type, utter_dim, conv_encoder,
                                      rnn_dropout, num_layers, dag_dropout, pooler_type)
    elif model_variant == 'csk':
        if csk_window > 0:
            model = CskCauseDag(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout,
                                num_layers, dag_dropout, pooler_type, add_emotion, emo_emb, emotion_dim)
        else:
            model = CskCauseDag0(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout,
                                 num_layers, dag_dropout, pooler_type, add_emotion, emo_emb, emotion_dim)
    elif model_variant == 'csk_trm':
        model = CskTransformer(model_size, utter_dim, conv_encoder, rnn_dropout, num_layers, nhead, ff_dim,
                               att_dropout, trm_dropout, max_len, pe_type, add_emotion, emo_emb, emotion_dim)
    else:
        raise NotImplementedError()

    loss_fn = MaskedBCELoss2()

    if multi_gpu:
        model = torch.nn.DataParallel(model)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.module.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.module.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    if scheduler_type == 'linear':
        num_conversations = len(train_loader.dataset)
        if (num_conversations * n_epochs) % (batch_size * accumulate_step) == 0:
            num_training_steps = (num_conversations * n_epochs) / (batch_size * accumulate_step)
        else:
            num_training_steps = (num_conversations * n_epochs) // (batch_size * accumulate_step) + 1
        num_warmup_steps = int(num_training_steps * args.warmup_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        scheduler = get_constant_schedule(optimizer)
    model = model.cuda()

    dev_fscore, test_fscore = train(model_variant, model, model_name, train_loader, dev_loader, test_loader,
                                    loss_fn, optimizer, n_epochs, log, accumulate_step, scheduler)
    return dev_fscore, test_fscore, model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', type=str, required=False, default='0')
    parser.add_argument('--n_epochs', type=int, required=False, default=40)
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--accumulate_step', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-3)
    parser.add_argument('--model_size', type=str, required=False, default='base')
    parser.add_argument('--model_variant', type=str, required=False, default='csk')
    parser.add_argument('--valid_shuffle', action='store_true', help='whether to shuffle dev and test sets')
    parser.add_argument('--scheduler', type=str, required=False, default='constant')
    parser.add_argument('--warmup_rate', type=float, required=False, default=0.1)
    parser.add_argument('--add_emotion', action='store_true')
    parser.add_argument('--emotion_dim', type=int, required=False, default=0)
    parser.add_argument('--dag_dropout', type=float, required=False, default=0.1)
    parser.add_argument('--pooler_type', type=str, required=False, default='all')
    parser.add_argument('--window', type=int, required=False, default=10)
    parser.add_argument('--csk_window', type=int, required=False, default=10)
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument('--trm_dropout', required=False, type=float, default=0.1)
    parser.add_argument('--att_dropout', required=False, type=float, default=0.1)
    parser.add_argument('--nhead', required=False, type=int, default=6)
    parser.add_argument('--ff_dim', required=False, type=int, default=600)
    parser.add_argument('--max_len', required=False, type=int, default=200)
    parser.add_argument('--pe_type', required=False, type=str, default='abs')

    parser.add_argument('--mapping_type', type=str, required=False, default='max')
    parser.add_argument('--utter_dim', type=int, required=False, default=300)
    parser.add_argument('--num_layers', type=int, required=False, default=1)
    parser.add_argument('--conv_encoder', type=str, required=False, default='none')
    parser.add_argument('--rnn_dropout', type=float, required=False, default=0.5)
    parser.add_argument('--seed', nargs='+', type=int, required=False, default=[0, 1, 2, 3, 4])
    parser.add_argument('--index', nargs='+', type=int, required=False, default=[1, 2, 3, 4, 5])

    parser.add_argument('--save_dir', type=str, required=False, default='saves')

    args_for_main = parser.parse_args()
    seed_list = args_for_main.seed
    index_list = args_for_main.index
    dev_fscore_list = []
    test_fscore_list = []
    model_dir = ''
    for sd, idx in zip(seed_list, index_list):
        dev_f1, test_f1, model_dir = main(args_for_main, sd, idx)
        dev_fscore_list.append(dev_f1)
        test_fscore_list.append(test_f1)

    dev_fscore_mean = np.round(np.mean(dev_fscore_list) * 100, 2)
    dev_fscore_std = np.round(np.std(dev_fscore_list) * 100, 2)

    test_fscore_mean = np.round(np.mean(test_fscore_list) * 100, 2)
    test_fscore_std = np.round(np.std(test_fscore_list) * 100, 2)

    logs_path = model_dir + '/log_metrics_' + str(index_list[0]) + '-' + str(index_list[-1]) + '.txt'
    logs = open(logs_path, 'w')

    logs.write(str(args_for_main) + '\n\n')

    log_lines = f'dev fscore: {dev_fscore_mean}(+-{dev_fscore_std})'
    print(log_lines)
    logs.write(log_lines + '\n')
    log_lines = f'test fscore: {test_fscore_mean}(+-{test_fscore_std})'
    print(log_lines)
    logs.write(log_lines + '\n')
    logs.close()
