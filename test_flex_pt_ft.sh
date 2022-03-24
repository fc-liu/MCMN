cuda=5

# Test TP-Train + Sup. Meta
pretrain_ckpt=wo-test-roberta-dropkeep-0.5-lr-5e-06-N-9_12000_84.6817
save_pref=pt_ft_test
bert_name=roberta
dropout=0.5
learn_rate=5e-06
N_num=8
save_full_name=$save_pref-$bert_name-dropkeep-$dropout-lr-$learn_rate-N-$N_num

meta_epochs=45


# finetune
CUDA_VISIBLE_DEVICES=$cuda python train.py --bert_model $bert_name --learning_rate $learn_rate --dropout_keep_prob $dropout  --N $N_num  --load_ckpt $pretrain_ckpt --save_ckpt $save_pref
#test
CUDA_VISIBLE_DEVICES=$cuda python test_flex.py --meta_epochs $meta_epochs --bert_model $bert_name --learning_rate $learn_rate --dropout_keep_prob $dropout --load_ckpt $save_full_name --test_batch_size 13 
fewshot score --for_leaderboard  --challenge_name flex --predictions output/predictions_$save_full_name.json
