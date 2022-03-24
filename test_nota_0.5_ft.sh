cuda=4

nota=0.5 #[0.15, 0.5]
bert_name=roberta
dropout=0.5
learn_rate=5e-06
N_num=5
ckpt_name=test_nota
ckpt_full_name=$ckpt_name-$bert_name-dropkeep-$dropout-lr-$learn_rate-N-$N_num-na-$nota
eta=5 

epoch=1
# Train model 
CUDA_VISIBLE_DEVICES=$cuda python train.py --na_rate $nota --bert_model $bert_name --dropout_keep_prob $dropout --learning_rate $learn_rate --N $N_num --save_ckpt $ckpt_name --batch_size 2

# evaluate 5-way 1-shot
K=1
echo $K-K $nota-nota $eta-eta :
test_file=data/fewrel_nota/data/test_wiki_input-5-${K}-${nota}.json
res_file=data/fewrel_nota/res/pred-5-${K}-${nota}.json
CUDA_VISIBLE_DEVICES=$cuda python test_nota_mcmn.py --nota_epoch $epoch --paral_cuda 0 --eta ${eta} --learning_rate 5e-6 --N 5 --K $K --test_file ${test_file} --load_ckpt $ckpt_full_name --na_rate ${nota} > ${res_file}


# evaluate 5-way 5-shot
K=5
echo $K-K $nota-nota $eta-eta :
test_file=data/fewrel_nota/data/test_wiki_input-5-${K}-${nota}.json
res_file=data/fewrel_nota/res/pred-5-${K}-${nota}.json
CUDA_VISIBLE_DEVICES=$cuda python test_nota_mcmn.py --nota_epoch $epoch --paral_cuda 0 --eta ${eta} --learning_rate 5e-6 --N 5 --K $K --test_file ${test_file} --load_ckpt $ckpt_full_name --na_rate ${nota} > ${res_file}
