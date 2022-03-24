from copy import deepcopy
from fewshot import make_challenge, Model
from typing import Any, Dict, Iterable, Sequence, Tuple, Union
from torch._C import import_ir_module
from configure import FLAGS
import random
import numpy as np
import torch
import torch.utils.data as data
from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer
import os
from dataloader import flex_dataloader
from tqdm import tqdm
from model.multi_choice import MultiChoiceNet
from torch.optim import Adam, lr_scheduler
import torch.nn as nn


evaluator = make_challenge("flex")
cost = nn.CrossEntropyLoss()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(FLAGS.seed)


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


class WrapperModel(Model):
    def __init__(self):
        super().__init__()

        # self.encoder = RobertaModel.from_pretrained(
        #     FLAGS.bert_model, output_attentions=True)

        # self.tokenizer = RobertaTokenizer.from_pretrained(
        #     FLAGS.bert_model, do_basic_tokenize=False)

        if 'roberta' in FLAGS.bert_model:
            self.encoder = RobertaModel.from_pretrained(
                FLAGS.bert_model, output_attentions=True)

            self.tokenizer = RobertaTokenizer.from_pretrained(
                FLAGS.bert_model, do_basic_tokenize=True)
        else:
            self.encoder = BertModel.from_pretrained(
                FLAGS.bert_model, output_attentions=True)

            self.tokenizer = BertTokenizer.from_pretrained(
                FLAGS.bert_model, do_basic_tokenize=True)

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [FLAGS.e11, FLAGS.e12, FLAGS.e21, FLAGS.e22, FLAGS.choice]})
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.model = MultiChoiceNet(self.encoder)
        self.model = self.model.to(FLAGS.paral_cuda[0])
        load_ckpt_file_path = "./checkpoint/fewrel/{}".format(FLAGS.load_ckpt)
        # save_ckpt_file_path = "./checkpoint/fewrel/{}".format(FLAGS.save_ckpt)
        self.model = nn.DataParallel(
            self.model, device_ids=FLAGS.paral_cuda)

        if os.path.exists(load_ckpt_file_path):
            print("######################load fewrel full model from {}#######################".format(
                load_ckpt_file_path))
            if FLAGS.paral_cuda[0] >= 0:
                ckpt = torch.load(load_ckpt_file_path, map_location=lambda storage,
                                  loc: storage.cuda(FLAGS.paral_cuda[0]))
            else:
                ckpt = torch.load(
                    load_ckpt_file_path, map_location=lambda storage, loc: storage.cpu())

            try:
                self.model.load_state_dict(ckpt["state_dict"])
            except Exception as e:
                self.model.module.load_state_dict(ckpt["state_dict"])

        self.orig_model_params = deepcopy(self.model.state_dict())

    def fit_and_predict(self,
                        support_x: Iterable[Any],
                        support_y: Iterable[str],
                        target_x: Iterable[Any],
                        metadata: Dict[str, Any] = None) -> Union[Sequence[str], Tuple[Sequence[str], Sequence[float]]]:
        ''''
        For zero-shot test
        '''
        is_train = False
        if len(support_x) > 0 and len(support_y) > 0:
            is_train = True
        dataset = flex_dataloader.MetaFewRelDataset(
            target_x, None, metadata, self.tokenizer)

        if is_train:
            train_dataset = flex_dataloader.MetaFewRelDataset(
                support_x, support_y, metadata, self.tokenizer)
            self.model.train()
            self.train(train_dataset)
        self.model.eval()
        predictions = self.test(dataset)
        scores = [1.0 for _ in predictions]
        if is_train:
            state_dict_to_load = deepcopy(self.orig_model_params)
            self.model.load_state_dict(state_dict_to_load)
        torch.cuda.empty_cache()
        return predictions, scores

    def train(self, train_dataset):
        self.model.train()
        # best_val_accu = 0.0
        parameters_to_optimize = self.model.parameters()
        optimizer = Adam(parameters_to_optimize, FLAGS.learning_rate,
                         weight_decay=FLAGS.l2_reg_lambda)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=FLAGS.lr_step_size, gamma=FLAGS.decay_rate)

        iter_loss = 1
        iter_sample = 1
        threshold = 0.05
        avg_acc = threshold+1
        avg_loss = threshold+1
        for _ in tqdm(range(FLAGS.meta_epochs)):
            iter_step = 0
        # while avg_loss < threshold and iter_step < 70:
            # for _ in range(FLAGS.meta_epochs):
            # for _ in range(20):
            data_loader = data.DataLoader(
                train_dataset, batch_size=FLAGS.test_batch_size, num_workers=4, pin_memory=True, shuffle=True, collate_fn=flex_dataloader.flex_collate_fn)
            # tq = tqdm(data_loader)
            iter_loss = 0
            iter_right = 0
            iter_sample = 0
            for iter_data in data_loader:
                batch_ins, batch_label, choice_idx = iter_data
                iter_sample += len(batch_label)
                try:
                    batch_label = batch_label.to(FLAGS.paral_cuda[0])
                except Exception as e:
                    print(e)
                batch_ins = [batch_ins['word'].to(FLAGS.paral_cuda[0]), batch_ins['pos1'].to(FLAGS.paral_cuda[0]),
                             batch_ins['pos2'].to(FLAGS.paral_cuda[0]), batch_ins['mask'].to(FLAGS.paral_cuda[0]), batch_ins['seg_ids'].to(FLAGS.paral_cuda[0])]

                logits, pred = self.model(
                    batch_ins, choice_idx, FLAGS.test_batch_size)
                loss = loss_fn(logits, batch_label)
                right = accuracy(pred, batch_label)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters_to_optimize, 5)
                optimizer.step()
                scheduler.step()
                iter_loss += loss.data.item()
                iter_right += right.data.item()*len(batch_label)
            avg_loss = iter_loss/iter_sample
            avg_acc = iter_right/iter_sample
            iter_step += 1
            # tq.set_postfix(loss=iter_loss / iter_sample)
            # cur_accu = iter_right/iter_sample
            # if cur_accu >

    def test(self, test_dataset):
        predictions = []
        self.model.eval()
        data_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=FLAGS.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=4,
                                      collate_fn=flex_dataloader.flex_collate_fn)
        # for batch_query, _, choice_idx in tqdm(data_loader):
        for batch_query, _, choice_idx in data_loader:
            batch_query = [batch_query['word'].to(FLAGS.paral_cuda[0]), batch_query['pos1'].to(FLAGS.paral_cuda[0]),
                           batch_query['pos2'].to(FLAGS.paral_cuda[0]), batch_query['mask'].to(FLAGS.paral_cuda[0]), batch_query['seg_ids'].to(FLAGS.paral_cuda[0])]

            _, pred = self.model(batch_query, choice_idx,
                                 FLAGS.test_batch_size)
            pred = pred.detach().cpu().tolist()
            predictions.extend(pred)
        return predictions


def main():
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)
    model = WrapperModel()
    evaluator = make_challenge("flex")
    evaluator.save_model_predictions(
        model=model,
        save_path="output/predictions_{}.json".format(FLAGS.load_ckpt),
        start_task_index=0,
        stop_task_index=None,
    )


if __name__ == "__main__":
    main()
