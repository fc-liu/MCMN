cuda=3

nota=0.15 #[0.15, 0.5]
bert_name=roberta
dropout=0.5
learn_rate=5e-06
N_num=5
ckpt_name=test_nota_pt_ft
ckpt_full_name=$ckpt_name-$bert_name-dropkeep-$dropout-lr-$learn_rate-N-$N_num-na-$nota
eta=1 
pretrain_ckpt=pretrain

epoch=2

# Train model 
CUDA_VISIBLE_DEVICES=$cuda python train.py --na_rate $nota --bert_model $bert_name --dropout_keep_prob $dropout --load_ckpt $pretrain_ckpt --learning_rate $learn_rate --N $N_num --save_ckpt $ckpt_name --batch_size 1

# # evaluate 5-way 1-shot
K=1
echo $K-K $nota-nota $eta-eta :
test_file=data/fewrel_nota/data/test_wiki_input-5-${K}-${nota}.json
res_file=data/fewrel_nota/res/pt-ft/pred-5-${K}-${nota}.json
CUDA_VISIBLE_DEVICES=$cuda python test_nota_mcmn.py --paral_cuda 0 --nota_epoch $epoch --learning_rate 5e-6 --N 5 --K $K --test_file ${test_file} --load_ckpt $ckpt_full_name --na_rate ${nota} > ${res_file}

# evaluate 5-way 5-shot
K=5
echo $K-K $nota-nota $eta-eta :
test_file=data/fewrel_nota/data/test_wiki_input-5-${K}-${nota}.json
res_file=data/fewrel_nota/res/pt-ft/pred-5-${K}-${nota}.json
CUDA_VISIBLE_DEVICES=$cuda python test_nota_mcmn.py --paral_cuda 0 --nota_epoch $epoch --learning_rate 5e-6 --N 5 --K $K --test_file ${test_file} --load_ckpt $ckpt_full_name --na_rate ${nota} > ${res_file}