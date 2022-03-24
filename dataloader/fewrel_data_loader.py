from cmd import PROMPT
import json
import os
import numpy as np
import random
import torch
import torch.utils.data as data
from configure import FLAGS
import sys
E11 = FLAGS.e11
E12 = FLAGS.e12
E21 = FLAGS.e21
E22 = FLAGS.e22

tokenizer = None


class BertEMDataset(data.Dataset):

    def __init__(self, file_name,  bert_tokenizer, N, K, random_cls_num=False, na_rate=FLAGS.na_rate, max_length=512):
        super(BertEMDataset, self).__init__()
        self.max_length = max_length
        self.bertTokenizer = bert_tokenizer
        global tokenizer
        tokenizer = self.bertTokenizer
        self.N = N
        self.K = K
        self.na_rate = na_rate
        self.random_cls_num = random_cls_num
        if not os.path.exists(file_name):
            raise Exception("[ERROR] Data file doesn't exist")
        if not os.path.exists(FLAGS.rel_name):
            raise Exception("[ERROR] Relation name file doesn't exist")

        self.json_data = json.load(open(file_name, "r"))
        # self.data = {}
        print("Finish loading file")
        self.classes = list(self.json_data.keys())
        self.id_to_name = json.load(open(FLAGS.rel_name))
        # self.id_to_name={name:name for name in self.classes} # in pubmed
        self.process_rel_name(self.id_to_name)
        self.__init_process_data__(self.json_data)
        # self.na_prompt=tokenizer.tokenize(FLAGS.na_prompt)
        self.na_prompt = FLAGS.na_prompt
        self.count = FLAGS.batch_size
        print("Finish init process data")

    def process_rel_name(self, name_dict):
        for key, val in name_dict.items():
            val.replace("_", " ")
            # val=tokenizer.tokenize(val)
            name_dict[key] = val

    def __init_process_data__(self, raw_data):
        def insert_and_tokenize(tokenizer, tokens, pos1, pos2, marker1, marker2):
            tokens.insert(pos2[-1]+1, marker2[-1])
            tokens.insert(pos2[0], marker2[0])
            tokens.insert(pos1[-1]+1, marker1[-1])
            tokens.insert(pos1[0], marker1[0])
            # tokens = tokens.copy()

            # tokens = tokenizer.tokenize(" ".join(tokens))

            return tokens

        for rel in self.classes:
            for ins in self.json_data[rel]:
                pos1 = ins['h'][2][0]
                pos2 = ins['t'][2][0]
                words = ins['tokens']

                if pos1[0] > pos2[0]:
                    tokens = insert_and_tokenize(self.bertTokenizer, words, pos2, pos1, [
                        E21, E22], [E11, E12])
                else:
                    tokens = insert_and_tokenize(self.bertTokenizer, words, pos1, pos2, [
                        E11, E12], [E21, E22])

                # pos1 = [tokens.index(FLAGS.e11), tokens.index(FLAGS.e12)]
                # pos2 = [tokens.index(FLAGS.e21), tokens.index(FLAGS.e22)]

                # if len(tokens) >= self.max_length:
                #     max_right = max(pos2[-1], pos1[-1])
                #     min_left = min(pos1[0], pos2[0])
                #     gap_length = max_right-min_left
                #     if gap_length+1 > self.max_length:
                #         tokens = [FLAGS.e11, FLAGS.e12,
                #                   FLAGS.e21, FLAGS.e22]
                #     elif max_right+1 < self.max_length:
                #         tokens = tokens[:self.max_length-1]
                #     else:
                #         tokens = tokens[min_left:max_right]

                ins["pos1"] = pos1
                ins['pos2'] = pos2
                ins["raw_tokens"] = ins['tokens']
                ins['tokens'] = tokens

                if len(tokens) > self.max_length:
                    raise Exception("sequence too long")

        print("init process data finish")

    def __additem__(self, d, instance):
        word = instance['tokens']
        pos1 = instance['pos1']
        pos2 = instance['pos2']
        prompt = instance['prompt']

        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['prompt'].append(prompt)

    def __getitem__(self, index):
        if self.random_cls_num:
            self.N = random.randint(5, 10)
        try:
            sampled_classes = random.sample(self.classes, self.N+1)
            target_classes = sampled_classes[:-1]
        except Exception:
            sampled_classes = random.sample(self.classes, self.N)
            target_classes = sampled_classes

        np.random.shuffle(target_classes)

        prompt_list = []
        for rel in target_classes:
            prompt_list.append(FLAGS.choice)
            try:
                prompt_list.append(self.id_to_name[rel])
            except KeyError:
                prompt_list.append(rel)
        if FLAGS.na_rate > 0:
            prompt_list.append(FLAGS.choice)
            prompt_list.append(self.na_prompt)

        # prompt_str = " ".join(prompt_list)
        # prompt = tokenizer.tokenize(prompt_str)
        prompt = prompt_list

        # taxo_dict=tokenizer(" ".join(taxonomy),add_special_tokens=False,return_tensors='pt',truncation=True,)

        support_set = {'word': [], 'pos1': [],
                       'pos2': [], 'mask': [], 'prompt': []}
        support_labels = np.arange(self.N)
        support_labels = np.repeat(support_labels, self.K).tolist()

        # shuffle label
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K, False)
            for j in indices:
                instance = self.json_data[class_name][j]
                instance["prompt"] = prompt
                # insert_prompt(instance,prompt)
                self.__additem__(support_set, instance)

        # NA
        if FLAGS.na_rate > 0:
            na_classes = sampled_classes[-1]
            Q_na = round(self.na_rate * self.N)
            cur_class = na_classes
            instances = np.random.choice(self.json_data[cur_class],
                                         Q_na, False)
            # instance = self.json_data[cur_class][index]
            # insert_prompt(instance,prompt)\
            for instance in instances:
                instance['prompt'] = prompt
                self.__additem__(support_set, instance)
            support_labels += [self.N] * Q_na

        # choice_idx=torch.tensor(choice_idx).long()
        return support_set, support_labels

    def __len__(self):
        return sys.maxsize


def idx_and_mask(batch_sets):
    global tokenizer
    # batch_sets = batch_sets.copy()
    # max_length = compute_max_length(batch_sets)
    sets = []
    batch_choice_idx = []
    batch_words = []
    batch_prompts = []
    set_item = {'word': [], 'pos1': [],
                'pos2': [], 'mask': [], 'seg_ids': []}
    for raw_set_item in batch_sets:
        words_list = raw_set_item['word']
        prompt_list = raw_set_item['prompt']
        batch_words.extend(words_list)
        batch_prompts.extend(prompt_list)
        # support_word = []
    tokens_dict = tokenizer(
        batch_prompts, batch_words, add_special_tokens=True, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=FLAGS.max_full_length, padding=True, return_token_type_ids=True)
    tokens_ids = tokens_dict['input_ids']

    mask = tokens_dict['attention_mask']
    set_item['mask'] = mask
    set_item['word'] = tokens_ids
    set_item['seg_ids'] = tokens_dict["token_type_ids"]
    # set_item.pop('prompt')
    for i, idx in enumerate(tokens_ids):
        cur_choice_idx = []
        tokens = tokenizer.convert_ids_to_tokens(idx)
        for j, token in enumerate(tokens):
            if token == FLAGS.choice:
                cur_choice_idx.append(j)

        pos1 = [tokens.index(FLAGS.e11), tokens.index(FLAGS.e12)]
        pos2 = [tokens.index(FLAGS.e21), tokens.index(FLAGS.e22)]
        if len(tokens) >= FLAGS.max_full_length:
            max_right = max(pos2[-1], pos1[-1])
            min_left = min(pos1[0], pos2[0])
            gap_length = max_right-min_left+cur_choice_idx[-1]
            if gap_length+1 > FLAGS.max_full_length:
                tokens = tokens[:cur_choice_idx[-1]]+[FLAGS.e11, FLAGS.e12,
                                                      FLAGS.e21, FLAGS.e22]
            elif max_right+1 < FLAGS.max_full_length:
                tokens = tokens[:FLAGS.max_full_length-1]
            else:
                tokens = tokens[:cur_choice_idx[-1]] + \
                    tokens[min_left:max_right]
            pos1 = [tokens.index(FLAGS.e11), tokens.index(FLAGS.e12)]
            pos2 = [tokens.index(FLAGS.e21), tokens.index(FLAGS.e22)]

        set_item['pos1'].append(torch.tensor(pos1))
        set_item['pos2'].append(torch.tensor(pos2))

        # sets.append(set_item)
        batch_choice_idx.append(cur_choice_idx)
    batch_choice_idx = torch.tensor(batch_choice_idx)
    return set_item, batch_choice_idx


def collate_fn(data):
    raw_support_sets, query_labels = zip(*data)
    batch_labels = []
    for labels in query_labels:
        batch_labels.extend(labels)
    # compute max length
    support_sets, choice_idx = idx_and_mask(raw_support_sets)

    # for i in range(len(support_sets)):
    #     for k in support_sets[i]:
    #         batch_support[k] += support_sets[i][k]
    # batch_label += query_labels[i]
    for k in ["pos1", "pos2"]:
        support_sets[k] = torch.stack(
            support_sets[k], 0)
    batch_labels = torch.tensor(batch_labels)

    return support_sets, batch_labels, choice_idx


def get_loader(file_path, tokenizer, N, K, batch_size, random_cls_num=False, max_length=FLAGS.max_sentence_length,
               num_workers=2):
    dataset = BertEMDataset(
        file_path, tokenizer, N, K, random_cls_num=random_cls_num, max_length=max_length, na_rate=FLAGS.na_rate)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
