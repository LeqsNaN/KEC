#!/usr/bin/env bash
python know_train.py --gpu_list='0' \
    --n_epochs=40 \
    --batch_size=4 \
    --accumulate_step=2 \
    --lr=3e-5 \
    --weight_decay=1e-4 \
    --model_size=base \
    --model_variant=csk \
    --valid_shuffle \
    --scheduler='constant' \
    --dag_dropout=0.0 \
    --pooler_type='all' \
    --window=2 \
    --csk_window=2 \
    --utter_dim=300 \
    --num_layers=5 \
    --conv_encoder='none' \
    --rnn_dropout=0.0 \
    --seed 0 1 2 3 4 \
    --index 1 2 3 4 5 \
    --save_dir=saves