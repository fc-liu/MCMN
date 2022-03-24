from tqdm.std import tqdm
from configure import FLAGS
from dataloader.pretrain_data_loader import get_pretrain_laoder, BertEMDataset
from dataloader.fewrel_data_loader import get_loader
from configure import FLAGS
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import os
from torch.optim import Adam, lr_scheduler
from model.multi_choice import MultiChoiceNet
import sys
import prettytable as pt


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


def validate(model,
             eval_iter,
             data_loader):
    '''
    model: a FewShotREModel instance
    B: Batch size
    N: Num of classes for each batch
    K: Num of instances for each class in the support set
    Q: Num of instances for each class in the query set
    eval_iter: Num of iterations
    ckpt: Checkpoint path. Set as None if using current model parameters.
    return: Accuracy
    '''
    # print("")
    model.eval()
    eval_dataset = data_loader

    iter_right = 0.0
    iter_sample = 0.0
    with torch.no_grad():
        for it in range(eval_iter):
            query, label, choice_idx = next(eval_dataset)
            # logits, pred = self.predict(
            #     model, support, query, B, N, K, Q, label)
            label = label.to(FLAGS.paral_cuda[0])
            query = [query['word'].to(FLAGS.paral_cuda[0]), query['pos1'].to(FLAGS.paral_cuda[0]),
                     query['pos2'].to(FLAGS.paral_cuda[0]), query['mask'].to(FLAGS.paral_cuda[0]), query['seg_ids'].to(FLAGS.paral_cuda[0])]

            _, pred = model(query, choice_idx, FLAGS.batch_size)
            # logit = logits.detach().cpu().numpy()
            right = accuracy(pred, label)
            iter_right += right.data.item()
            iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(
                it + 1, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
        print("")
        accu = iter_right / iter_sample
        # print("####################accuracy: %.4f#####################" % accu)
    return accu


# bert_model = RobertaModel.from_pretrained(
#     FLAGS.bert_model, output_attentions=True)

# tokenizer = RobertaTokenizer.from_pretrained(
#     FLAGS.bert_model, do_basic_tokenize=False)

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

load_ckpt_file_path = "./checkpoint/pretrain_new/{}".format(FLAGS.load_ckpt)
save_ckpt_file_path="./checkpoint/pretrain_new/{}".format(FLAGS.save_ckpt)
print("#######################################")
print(load_ckpt_file_path)

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
    model.load_state_dict(ckpt["state_dict"])
    print("######################load fewrel full model from {}#######################".format(
        load_ckpt_file_path))

parameters_to_optimize = model.parameters()
optimizer = Adam(parameters_to_optimize, FLAGS.learning_rate,
                 weight_decay=FLAGS.l2_reg_lambda)
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=FLAGS.lr_step_size, gamma=FLAGS.decay_rate)

dataset = BertEMDataset(
    "pretrain_data_process/pretrain.json", tokenizer, max_length=FLAGS.max_sentence_length)

N_val = 5

eval_data_laoder = get_loader(
    './data/val_flex.json', tokenizer, N_val, FLAGS.K, FLAGS.batch_size, num_workers=4)
test_data_loader = get_loader(
    './data/val_flex.json', tokenizer, N_val, FLAGS.K, FLAGS.batch_size, num_workers=4)
eval_iter = 500
table = pt.PrettyTable(
    ["step", "val", "test"])
model.train()
best_acc = 0

for epoch in range(FLAGS.epochs):
    start_iter = 0
    iter_loss = 0.0
    iter_right = 0.0
    iter_sample = 0.0

    pretrain_dataloader = get_pretrain_laoder(dataset, FLAGS.pretrain_batch_size)
    tq = tqdm(pretrain_dataloader)
    for start_iter, iter_data in enumerate(tq):
        batch_ins, batch_label, choice_idx = iter_data
        batch_label = batch_label.to(FLAGS.paral_cuda[0])
        batch_ins = [batch_ins['word'].to(FLAGS.paral_cuda[0]), batch_ins['pos1'].to(FLAGS.paral_cuda[0]),
                     batch_ins['pos2'].to(FLAGS.paral_cuda[0]), batch_ins['mask'].to(FLAGS.paral_cuda[0]), batch_ins['seg_ids'].to(FLAGS.paral_cuda[0])]
        logits, pred = model(batch_ins, choice_idx, FLAGS.pretrain_batch_size)
        loss = loss_fn(logits, batch_label)
        right = accuracy(pred, batch_label)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters_to_optimize, 5)
        optimizer.step()
        scheduler.step()
        iter_loss += loss.data.item()
        iter_right += right.data.item()
        iter_sample += 1
        # sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(
        #     start_iter + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
        # sys.stdout.flush()
        tq.set_postfix(loss=iter_loss / iter_sample,
                       accuracy=100 * iter_right / iter_sample)

        if start_iter % 1000 == 0:
            acc_val = validate(model, eval_iter, eval_data_laoder)
            acc_test = validate(model, eval_iter, test_data_loader)
            table.add_row(
                [start_iter, round(100*acc_val, 4), round(100*acc_test, 4)])
            print(table)
            model.train()
            if acc_val+acc_test > best_acc:
                print('Best checkpoint')
                best_acc = acc_val+acc_test
                torch.save({'state_dict': model.module.state_dict()},
                           os.path.join(save_ckpt_file_path+"_"+str(start_iter)+"_"+str(round(100*best_acc/2, 4))))

            print(
                "#################best eval accu: %.4f##################" % (best_acc/2))

        # start_iter+=1
