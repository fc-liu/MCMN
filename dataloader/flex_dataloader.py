import torch.utils.data as data
from configure import FLAGS
import torch
import numpy as np
tokenizer = None


class MetaFewRelDataset(data.Dataset):
    def __init__(self, batch_data, batch_label, metadata, lm_tokenizer):
        self.data_list = []
        self.batch_label = batch_label
        self.tokenizer = lm_tokenizer
        global tokenizer
        tokenizer = self.tokenizer
        self.data_list = self.init_process(
            batch_data, metadata)
        self.prompt_tokens = self.get_promtp_tokens(metadata)

    def get_promtp_tokens(self, metadata):
        text_label_dict = metadata['text_labels']
        text_label_list = sorted(text_label_dict.items(), key=lambda x: x[1])
        prompt_list = [i[0] for i in text_label_list]
        prompt_token_list = []
        for prompt in prompt_list:
            prompt_token_list.append(FLAGS.choice)
            prompt_token_list.append(prompt)
        # prompt_str = " ".join(prompt_token_list)
        # prompt_tokens = self.tokenizer.tokenize(prompt_str)
        # return prompt_tokens

        return prompt_token_list

    def init_process(self, batch_data, metadata):
        def replace_marker(text):
            text = text.replace("* ", FLAGS.e11+" ", 1)
            text = text.replace(" *", " "+FLAGS.e12, 1)
            text = text.replace("# ", FLAGS.e21+" ", 1)
            text = text.replace(" #", " "+FLAGS.e22, 1)
            return text
        ins_list = []

        for raw_ins in batch_data:
            ins = {}
            txt = raw_ins['txt']
            txt = replace_marker(txt)
            # tokens = self.tokenizer.tokenize(txt)
            # ins['tokens'] = tokens

            ins['tokens'] = txt

            # pos1 = [tokens.index(FLAGS.e11), tokens.index(FLAGS.e12)]
            # pos2 = [tokens.index(FLAGS.e21), tokens.index(FLAGS.e22)]
            # if len(tokens) >= FLAGS.max_sentence_length:
            #     max_right = max(pos2[-1], pos1[-1])
            #     min_left = min(pos1[0], pos2[0])
            #     gap_length = max_right-min_left
            #     if gap_length+1 > FLAGS.max_sentence_length:
            #         tokens = [FLAGS.e11, FLAGS.e12,
            #                   FLAGS.e21, FLAGS.e22]
            #     elif max_right+1 < FLAGS.max_sentence_length:
            #         tokens = tokens[:FLAGS.max_sentence_length-1]
            #     else:
            #         tokens = tokens[min_left:max_right]

            # ins["raw_tokens"] = ins['tokens']
            # ins['tokens'] = tokens
            # ins['prompt'] = predicate

            ins_list.append(ins)

        return ins_list

    def __getitem__(self, index):
        label = self.batch_label
        if self.batch_label:
            label = int(self.batch_label[index])
        return self.data_list[index], label, self.prompt_tokens

    def __len__(self):
        return len(self.data_list)


def idx_and_mask(raw_ins_list, batch_prompts):
    global tokenizer
    # batch_sets = batch_sets.copy()
    # max_length = compute_max_length(batch_sets)
    sets = []
    batch_choice_idx = []
    set_item = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'seg_ids': []}

    batch_tokens = []
    for raw_ins in raw_ins_list:
        batch_tokens.append(raw_ins['tokens'])

    tokens_dict = tokenizer(
        batch_prompts, batch_tokens, add_special_tokens=True, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=FLAGS.max_full_length, padding=True, return_token_type_ids=True)
    tokens_ids = tokens_dict['input_ids']

    mask = tokens_dict['attention_mask']
    set_item['mask'] = mask
    set_item['word'] = tokens_ids
    set_item['seg_ids'] = tokens_dict["token_type_ids"]
    # set_item.pop('prompt')
    for idx in tokens_ids:
        cur_choice_idx = []
        tokens = tokenizer.convert_ids_to_tokens(idx)
        # if len(cur_choice_idx) == 0:
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
            print(f"prompt lens:{len(batch_prompts[0])}")
            print(f"tokens shape:{np.asarray(tokens).shape}")
        batch_choice_idx.append(cur_choice_idx)
        set_item['pos1'].append(torch.tensor(pos1))
        set_item['pos2'].append(torch.tensor(pos2))

    # sets.append(set_item)

    batch_choice_idx = torch.tensor(batch_choice_idx)
    return set_item, batch_choice_idx


def flex_collate_fn(data):

    raw_support_sets, batch_labels, prompt_list = zip(*data)

    # compute max length
    support_sets, choice_idx = idx_and_mask(raw_support_sets, prompt_list)

    # for i in range(len(support_sets)):
    #     for k in support_sets[i]:
    #         batch_support[k] += support_sets[i][k]
    # batch_label += query_labels[i]
    for k in ["pos1", "pos2"]:
        support_sets[k] = torch.stack(
            support_sets[k], 0)

    if batch_labels[0] != None:
        # for i in range(len(label_idx)):
        #     label_idx[i] = int(label_idx[i])
        batch_labels = torch.tensor(batch_labels)
    return support_sets, batch_labels, choice_idx


def get_loader(dataset, batch_size,
               num_workers=4):

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=flex_collate_fn)
    return iter(data_loader)
