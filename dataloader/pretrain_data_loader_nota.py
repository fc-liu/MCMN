import random
import json
import os
import numpy as np
import torch
import torch.utils.data as data
from configure import FLAGS
from tqdm import tqdm
E11 = FLAGS.e11
E12 = FLAGS.e12
E21 = FLAGS.e21
E22 = FLAGS.e22
tokenizer = None


class BertEMDataset(data.Dataset):

    def __init__(self, file_name, bert_tokenizer, max_length=512):
        super(BertEMDataset, self).__init__()
        self.max_length = max_length
        self.bertTokenizer = bert_tokenizer
        global tokenizer
        tokenizer = self.bertTokenizer

        if not os.path.exists(file_name):
            raise Exception("[ERROR] Data file doesn't exist")

        # self.data = {}
        print("Finish loading file")
        self.json_data, self.pred_list = self.__init_process_data__(file_name)
        # self.na_prompt=tokenizer.tokenize(FLAGS.na_prompt)
        self.na_prompt = FLAGS.na_prompt
        print("Finish init process data")

    def __init_process_data__(self, file_name):
        def insert_and_tokenize(tokenizer, tokens, pos1, pos2, marker1, marker2):
            tokens.insert(pos2[-1]+1, marker2[-1])
            tokens.insert(pos2[0], marker2[0])
            tokens.insert(pos1[-1]+1, marker1[-1])
            tokens.insert(pos1[0], marker1[0])
            # tokens = tokens.copy()

            # tokens = tokenizer.tokenize(" ".join(tokens))
            return tokens

        ret_data = []
        pred_list = []
        with open(file_name, 'r') as file:
            # i=0
            for json_line in tqdm(file.readlines()):
                # i+=1
                # if i>10:
                #     break
                full_ins = json.loads(json_line)
                predicate = full_ins['pred']
                predicate = predicate
                pred_list.append(predicate)

                ins = full_ins['gen']  # currently only use the generated text
                if len(ins) > 0:
                    ins = ins[0]
                else:
                    continue

                pos1 = ins['h'][2]
                pos2 = ins['t'][2]
                words = ins['tokens']

                if pos1[0] > pos2[0]:
                    tokens = insert_and_tokenize(self.bertTokenizer, words, pos2, pos1, [
                        E21, E22], [E11, E12])
                else:
                    tokens = insert_and_tokenize(self.bertTokenizer, words, pos1, pos2, [
                        E11, E12], [E21, E22])

                ins["raw_tokens"] = ins['tokens']
                ins['tokens'] = tokens
                ins['prompt'] = predicate

                ret_data.append(ins)

        return ret_data, pred_list

    def __getitem__(self, index):

        ins = self.json_data[index]

        return ins, ins['prompt']

    def __len__(self):
        return len(self.json_data)


def idx_and_mask(raw_ins_list, prompt_tokens):
    global tokenizer
    # batch_sets = batch_sets.copy()
    # max_length = compute_max_length(batch_sets)
    sets = []
    batch_choice_idx = []
    set_item = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'seg_ids': []}

    batch_tokens = []
    batch_promtps = []
    for raw_ins in raw_ins_list:
        batch_tokens.append(raw_ins['tokens'])
        batch_promtps.append(prompt_tokens)

    tokens_dict = tokenizer(
        batch_promtps, batch_tokens, add_special_tokens=True, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=FLAGS.max_full_length, padding=True, return_token_type_ids=True)
    tokens_ids = tokens_dict['input_ids']

    mask = tokens_dict['attention_mask']
    set_item['mask'] = mask
    set_item['word'] = tokens_ids
    set_item['seg_ids'] = tokens_dict["token_type_ids"]
    # set_item.pop('prompt')
    cur_choice_idx = []
    for idx in tokens_ids:
        tokens = tokenizer.convert_ids_to_tokens(idx)
        if len(cur_choice_idx) == 0:
            for j, token in enumerate(tokens):
                if token == FLAGS.choice:
                    cur_choice_idx.append(j)
        try:
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
        except ValueError as e:
            print(e)
            print("#################################")
            print(f"prompt lens:{len(batch_promtps[0])}")
            print(f"tokens shape:{np.asarray(tokens).shape}")
        set_item['pos1'].append(pos1)
        set_item['pos2'].append(pos2)
    set_item['pos1'] = torch.tensor(set_item['pos1'])
    set_item['pos2'] = torch.tensor(set_item['pos2'])
    sets.append(set_item)
    batch_choice_idx.append(cur_choice_idx)

    batch_choice_idx = torch.tensor(batch_choice_idx)
    return sets, batch_choice_idx


def pretrain_collate_fn(data):
    batch_query = {'word': [], 'pos1': [],
                   'pos2': [], 'mask': [], 'seg_ids': []}
    batch_label = []
    raw_ins_list, prompt_list = zip(*data)

    label_idx = []
    # np.random.shuffle(label_idx)
    # prompt_list=np.asarray(prompt_list)

    # prompt_list=prompt_list[label_idx]
    # prompt_list=list(prompt_list)
    prompt_token_list = []
    idx = 0
    label_desp = []
    label_num = 0
    for prompt in prompt_list:
        p = 0.5
        p1 = random.random()
        if p1 < p:
            label_desp.append(FLAGS.na_prompt)
        else:
            label_desp.append(prompt)
            label_num += 1

    for desp in label_desp:
        if desp == FLAGS.na_prompt:
            label_idx.append(label_num)
        else:
            prompt_token_list.append(FLAGS.choice)
            prompt_token_list.append(desp)
            label_idx.append(idx)
            idx += 1
    prompt_token_list.append(FLAGS.choice)
    prompt_token_list.append(FLAGS.na_prompt)
    # prompt_str = " ".join(prompt_token_list)
    # prompt_tokens = tokenizer.tokenize(prompt_str)
    prompt_tokens = prompt_token_list

    # compute max length
    query_sets, choice_idx = idx_and_mask(raw_ins_list, prompt_tokens)

    for i in range(len(query_sets)):
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
    for k in batch_query:
        batch_query[k] = torch.stack(
            batch_query[k], 0)
    batch_label = torch.tensor(label_idx)

    return batch_query, batch_label, choice_idx


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_pretrain_laoder(dataset, batch_size,
                        num_workers=4):

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=pretrain_collate_fn)
    return iter(data_loader)
