cuda=7

CUDA_VISIBLE_DEVICES=$cuda python pretrain_nota.py --pretrain_batch_size 10 --learning_rate 5e-6 --dropout_keep_prob 0.5 --bert_model roberta --save_ckpt 'pretrain_nota' --na_rate 0.5