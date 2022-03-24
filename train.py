from model.relation_representation_model import EntityMarkerEncoder, EntityMarkerClsEncoder
from configure import FLAGS
# import sklearn.exceptions
from dataloader.fewrel_data_loader import get_loader
# from dataloader.flex_dataloader import get_loader
from framework import FewShotREFramework
import sys
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from torch import nn
from torch.optim import Adam
import os
import torch
# from model.interact_proto import GlobalTransformedProtoNet_proto_three, GlobalTransformedProtoNet_three, GlobalTransformedProtoNet_new, InstanceTransformer, InteractiveContrastiveNet, GlobalTransformedProtoNet, Proto, ProtoHATT, GlobalTransformedProtoNet_onehot, GlobalTransformedProtoNet_all_query, GlobalTransformedProtoNet_proto_tag, GlobalTransformedProtoNet_proto_tag_cos
from model.multi_choice import MultiChoiceNet
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

seed = FLAGS.seed


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

fc_out_size = 256

relation_encoder = None
if FLAGS.rel_rep_model == 'em':
    relation_encoder = EntityMarkerEncoder()
elif FLAGS.rel_rep_model == "emc":
    relation_encoder = EntityMarkerClsEncoder()
else:
    raise NotImplementedError

if 'roberta' in FLAGS.bert_model:
    bert_model = RobertaModel.from_pretrained(
        FLAGS.bert_model, output_attentions=True)

    tokenizer = RobertaTokenizer.from_pretrained(
        FLAGS.bert_model, do_basic_tokenize=True)
else:
    bert_model = BertModel.from_pretrained(
        FLAGS.bert_model, output_attentions=True)

    tokenizer = BertTokenizer.from_pretrained(
        FLAGS.bert_model, do_basic_tokenize=True)

tokenizer.add_special_tokens(
    {"additional_special_tokens": [FLAGS.e11, FLAGS.e12, FLAGS.e21, FLAGS.e22, FLAGS.choice]})
bert_model.resize_token_embeddings(len(tokenizer))

load_ckpt_file_path = "./checkpoint/fewrel/{}".format(FLAGS.load_ckpt)
save_ckpt_file_path = "./checkpoint/fewrel/{}".format(FLAGS.save_ckpt)
print("#######################################")
print(load_ckpt_file_path)

print("{}-way-{}-shot Few-Shot Relation Classification".format(FLAGS.N, FLAGS.K))

max_length = FLAGS.max_sentence_length

# if FLAGS.zero_shot:
#     dk=0
# else:

N_train = FLAGS.N
N_val = 5


gpu_aval = torch.cuda.is_available()


model = MultiChoiceNet(bert_model)
model = model.to(FLAGS.paral_cuda[0])

model = nn.DataParallel(
    model, device_ids=FLAGS.paral_cuda)

if os.path.exists(load_ckpt_file_path):
    if FLAGS.paral_cuda[0] >= 0:
        ckpt = torch.load(load_ckpt_file_path, map_location=lambda storage,
                          loc: storage.cuda(FLAGS.paral_cuda[0]))
    else:
        ckpt = torch.load(
            load_ckpt_file_path, map_location=lambda storage, loc: storage.cpu())
    try:
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        model.module.load_state_dict(ckpt["state_dict"])
    print("######################load fewrel full model from {}#######################".format(
        load_ckpt_file_path))


if FLAGS.na_rate > 0:
    train_data_loader = get_loader(
        './data/train.json', tokenizer, N_train, FLAGS.K, FLAGS.batch_size)

    val_data_loader = get_loader(
        './data/val.json', tokenizer, N_val, FLAGS.K, FLAGS.batch_size)
    val2_data_loader = get_loader(
        './data/val.json', tokenizer, N_val, FLAGS.K, FLAGS.batch_size)
else:
    train_data_loader = get_loader(
        './data/train_flex.json', tokenizer, N_train, FLAGS.K, batch_size=FLAGS.batch_size, random_cls_num=True)

    val_data_loader = get_loader(
        './data/val_flex.json', tokenizer, N_val, FLAGS.K, FLAGS.batch_size)
    val2_data_loader = get_loader(
        './data/val_flex.json', tokenizer, N_val, FLAGS.K, FLAGS.batch_size)



framework = FewShotREFramework(
    train_data_loader, val_data_loader, val2_data_loader)

framework.train(model, FLAGS.save_ckpt, FLAGS.batch_size, N_train, N_val, FLAGS.K,
                learning_rate=FLAGS.learning_rate, weight_decay=FLAGS.l2_reg_lambda, optimizer=Adam, ckpt_file=save_ckpt_file_path)
