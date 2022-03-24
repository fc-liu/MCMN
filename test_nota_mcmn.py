from transformers.utils.dummy_pt_objects import BertModel
from configure import FLAGS
from dataloader.eval_dataloader_multi_choice import get_loader
from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
from torch import nn
from torch.optim import Adam, lr_scheduler
import os
import torch
import torch.utils.data as data
import prettytable as pt
from model.multi_choice import MultiChoiceNet
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import random

N = FLAGS.N
K = FLAGS.K


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(FLAGS.seed)


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

ckpt_file_path = "./checkpoint/fewrel/{}".format(FLAGS.load_ckpt)

max_length = FLAGS.max_sentence_length


gpu_aval = torch.cuda.is_available()

model = MultiChoiceNet(bert_model)
model = model.to(FLAGS.paral_cuda[0])

model = nn.DataParallel(
    model, device_ids=FLAGS.paral_cuda)

if os.path.exists(ckpt_file_path):
    if FLAGS.paral_cuda[0] >= 0:
        ckpt = torch.load(ckpt_file_path, map_location=lambda storage,
                          loc: storage.cuda(FLAGS.paral_cuda[0]))
    else:
        ckpt = torch.load(
            ckpt_file_path, map_location=lambda storage, loc: storage.cpu())
    try:
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        model.module.load_state_dict(ckpt["state_dict"])


# model=nn.DataParallel(model,device_ids=FLAGS.paral_cuda)

test_data_loader = get_loader(
    FLAGS.test_file, tokenizer, num_workers=4)

cost = nn.CrossEntropyLoss()


def loss_fn(logits, label):
    '''
    logits: Logits with the size (..., class_num)
    label: Label with whatever size.
    return: [Loss] (A single value)
    '''
    N = logits.size(-1)
    return cost(logits.view(-1, N), label.view(-1))


def accuracy(pred, label):
    '''
    pred: Prediction results with whatever size
    label: Label with whatever size
    return: [Accuracy] (A single value)
    '''
    return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


def fit(model, support_set, choice_idx):
    # best_val_accu = 0.0
    model.train()

    parameters_to_optimize = model.parameters()
    optimizer = Adam(parameters_to_optimize, FLAGS.learning_rate,
                     weight_decay=FLAGS.l2_reg_lambda)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=FLAGS.lr_step_size, gamma=FLAGS.decay_rate)

    iter_sample = support_set['word'].shape[0]
    iter_loss = iter_sample

    support = [support_set['word'].to(FLAGS.paral_cuda[0]), support_set['pos1'].to(FLAGS.paral_cuda[0]),
               support_set['pos2'].to(FLAGS.paral_cuda[0]), support_set['mask'].to(FLAGS.paral_cuda[0]), support_set['seg_ids'].to(FLAGS.paral_cuda[0])]
    batch_label = torch.arange(N)
    batch_label = batch_label.expand(K, -1).permute(1, 0).reshape(-1)
    batch_label = batch_label.to(FLAGS.paral_cuda[0])
    # for _ in tqdm(range(FLAGS.meta_epochs)):

    # while iter_loss/iter_sample > FLAGS.eta:
    for _ in range(FLAGS.nota_epoch):
        # for _ in range(4):
        iter_loss = 0
        iter_right = 0

        logits, pred = model(support, choice_idx, FLAGS.N*(FLAGS.K+1))
        loss = loss_fn(logits, batch_label)
        right = accuracy(pred, batch_label)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters_to_optimize, 5)
        optimizer.step()
        scheduler.step()
        iter_loss += loss.data.item()
        iter_right += right.data.item()
        # tq.set_postfix(loss=iter_loss / iter_sample)


orig_model_params = deepcopy(model.state_dict())


def test(model, data_loader):
    res = []
    for support, query, choice_idx in tqdm(data_loader):

        fit(model, support, choice_idx)
        model.eval()
        query = [query['word'].to(FLAGS.paral_cuda[0]), query['pos1'].to(FLAGS.paral_cuda[0]),
                 query['pos2'].to(FLAGS.paral_cuda[0]), query['mask'].to(FLAGS.paral_cuda[0]), query['seg_ids'].to(FLAGS.paral_cuda[0])]
        _, pred = model(query, choice_idx, N+1)
        # norm_logits = softmax(logits).detach().cpu().numpy().tolist()
        pred = pred.detach().cpu().item()
        # if max(logits[0][0]) < -1200:
        #     pred = -1
        if pred == len(choice_idx[0])-1:
            pred = -1
        state_dict_to_load = deepcopy(orig_model_params)
        model.load_state_dict(state_dict_to_load)
        # torch.cuda.empty_cache()
        res.append(pred)

    return res


# class Evaluator:
#     def __init__(self):


res = test(model,  data_loader=test_data_loader)
print(res)
