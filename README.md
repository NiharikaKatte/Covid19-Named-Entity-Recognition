# Covid19-Named-Entity-Recognition

To train use below command:
```shell
! python ner.py \
    --data_dir=<path_to_data_directory> \
    --glove_txt_file=<path_to_glove>\
    --output_dir=<path_to_output_directory> \
    --seed 2021 \
    --do_train True \
    --do_eval True \
    --do_test False \
    --max_seq_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 1e-3 \
    --num_train_epochs 50 \
    --evaluate_steps 100 \
    --rnn_type="gru" \
    --rnn_layers 2 \
    --rnn_dim 128 \
    --rnn_dropout 0.0 \
    --patience 3
```

To predict use below commad:
```shell
! python ner.py \
    --data_dir=<path_to_data_directory> \
    --glove_txt_file=<path_to_glove>\
    --output_dir=<path_to_output_directory> \
    --checkpoint=<checkpoint_path> \
    --seed 2021 \
    --do_train True \
    --do_eval True \
    --do_test False \
    --max_seq_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 1e-3 \
    --num_train_epochs 50 \
    --evaluate_steps 100 \
    --rnn_type="gru" \
    --rnn_layers 2 \
    --rnn_dim 128 \
    --rnn_dropout 0.0 \
    --patience 3
```