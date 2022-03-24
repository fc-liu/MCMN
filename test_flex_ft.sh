cuda=6

# Test Sup. Meta
# finetune
save_pref_ft=wo_pt_rand
bert_name=roberta
dropout=0.5
learn_rate=5e-06
N_num=8
save_full_name_ft=$save_pref_ft-$bert_name-dropkeep-$dropout-lr-$learn_rate-N-$N_num

meta_epochs=45
# finetune
CUDA_VISIBLE_DEVICES=$cuda python train.py --bert_model $bert_name --learning_rate $learn_rate --dropout_keep_prob $dropout  --N $N_num  --save_ckpt $save_pref_ft
#test
CUDA_VISIBLE_DEVICES=$cuda python test_flex.py --meta_epochs $meta_epochs --bert_model $bert_name --learning_rate $learn_rate --dropout_keep_prob $dropout --load_ckpt $save_full_name_ft --test_batch_size 13 

echo Sup. Meta
fewshot score --for_leaderboard  --challenge_name flex --predictions output/predictions_$save_full_name_ft.json
