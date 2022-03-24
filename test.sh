val_file="val_pubmed.json"
for N in 5 10;do
    for K in 1 5;do
        echo ${N}-way ${K}-shot:
        for seed in 1 12 123 1234 12345;do
            python sample_io.py data/${val_file} 100 $N $K $seed input > data/sample.json
            python sample_io.py data/${val_file} 100 $N $K $seed output > data/ans.json
            # CUDA_VISIBLE_DEVICES=5 python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --ckpt_name gtp-woseg-sepexcludproto-lay1-head1 --layer 1 --n_head 1 --model_name gtp --paral_cuda 0 > data/res.json
            # CUDA_VISIBLE_DEVICES=4 python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --ckpt_name bert-em --model_name proto --paral_cuda 0 > data/res.json
            CUDA_VISIBLE_DEVICES=6 python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --ckpt_name three_inter --abla three_inter --layer 2 --n_head 4 --model_name proto_three --paral_cuda 0 > data/res.json
            python evaluate.py data/res.json data/ans.json
        done
    done
done