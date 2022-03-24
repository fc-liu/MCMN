import torch.nn as nn
import torch
from configure import FLAGS
import framework
from model.relation_representation_model import RRModel


def extract_position(x, pos):
    bsz = x.shape[0]
    emb = x[torch.arange(bsz), pos[:, 0]]
    # e2=x[torch.arange(bsz),e[:,1]]
    # e_emb=torch.cat((e1,e2),dim=1)
    return emb


class MultiChoiceNet(framework.FewShotREModel):

    def __init__(self, embedder):
        super(MultiChoiceNet, self).__init__(None)
        self.pretrained_encoder = embedder
        self.hidden_size = FLAGS.hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.distance = nn.PairwiseDistance(p=2)

    def __dist__(self, cls_emb, q_emb):
        # cls_emb: B, cls_num, dim
        # q_emb: B, dim
        _, cls_num, _ = cls_emb.shape
        q_emb = q_emb.unsqueeze_(1).expand(-1, cls_num, -1)

        # dists = torch.pow(cls_emb-q_emb, 2).sum(2)
        dists = self.distance(cls_emb, q_emb)

        return dists

    def few_shot_match(self, support_embs, query_embs, choice_idx, s_pos1, s_pos2, q_pos1, q_pos2):
        # support_embs: bsz, n, k, seq_len, hidden_size
        # query_embs:bsz, total_Q, seq_len, hidden_size
        bsz, n, k, seq_len, dim = support_embs.shape
        support_embs = support_embs.view(bsz*n*k, seq_len, dim)
        _, total_Q, seq_len_q, dim = query_embs.shape
        query_embs = query_embs.view(-1, seq_len_q, dim)

        sup_head_emb = extract_position(support_embs, s_pos1)
        sup_tail_emb = extract_position(support_embs, s_pos2)
        sup_ins_rep = torch.cat([sup_head_emb, sup_tail_emb], dim=-1)
        query_head_emb = extract_position(query_embs, q_pos1)
        query_tail_emb = extract_position(query_embs, q_pos2)
        query_ins_rep = torch.cat([query_head_emb, query_tail_emb], dim=-1)

        sup_ins_rep = sup_ins_rep.view(bsz, n, k, 2*dim)
        sup_cls_rep = torch.mean(sup_ins_rep, 2)  # bsz, n, 2*dim

        query_ins_rep = query_ins_rep.view(-1, 2*dim)  # bsz*n*total_Q, 2*dim

        logits = -self.__dist__(sup_cls_rep, query_ins_rep)

        _, pred = torch.max(logits, 1)

        return logits, pred

    def test_choice_embs(self, query_embs, choice_embs, choice_idx):
        query_list = query_embs.detach().cpu().numpy().tolist()
        choice_list = choice_embs.detach().cpu().numpy().tolist()
        idx_list = choice_idx.detach().cpu().numpy().tolist()

        for i, ins_idx in enumerate(idx_list):
            for j, idx in enumerate(ins_idx):
                assert query_list[i][idx] == choice_list[i][j]

    def test_head_tail_embs(self, query_embs, head_emb, tail_emb, head_pos, tail_pos):
        query_list = query_embs.detach().cpu().numpy().tolist()
        head_list = head_emb.detach().cpu().numpy().tolist()
        tail_list = tail_emb.detach().cpu().numpy().tolist()
        for idx in range(len(head_pos)):
            pos1 = head_pos[idx][0]
            assert query_list[idx][pos1] == head_list[idx]

            pos2 = tail_pos[idx][0]
            assert query_list[idx][pos2] == tail_list[idx]

    def zero_shot_match(self, query_embs, choice_idx, pos1, pos2):
        # query_embs: batch_size, sequence_lens, hidden_size
        # choice_idx: batch_size, label_size

        # n, seq_len, h_dim = query_embs.shape
        choice_idx = choice_idx.transpose(1, 0)

        input_shape = query_embs.shape

        choice_embs = query_embs[torch.arange(
            input_shape[0]), choice_idx].transpose(1, 0)  # n, cls_num, hidden_size

        choice_idx = choice_idx.transpose(1, 0)

        head_rep = extract_position(query_embs, pos1)  # n, hidden_size
        tail_rep = extract_position(query_embs, pos2)  # n, hidden_size

        # self.test_choice_embs(query_embs, choice_embs, choice_idx)
        # self.test_head_tail_embs(query_embs, head_rep, tail_rep, pos1, pos2)

        ins_rel_rep = (head_rep+tail_rep)/2  # n, hidden_size
        # ins_rel_rep=head_rep-tail_rep
        ins_rel_rep = self.layer_norm(ins_rel_rep)  # n, hidden_size

        logits = -self.__dist__(choice_embs, ins_rel_rep)
        _, pred = torch.max(logits, 1)
        return logits, pred

    def forward(self, batch_data, choice_idx, batch_size):
        '''
        batch_data: Inputs of the batch data.
        choice_idx: choice token position
        batch_size: batch size
        '''
        # tokens, pos1, pos2, mask
        batch_embs = self.pretrained_encoder(
            input_ids=batch_data[0], attention_mask=batch_data[3], token_type_ids=batch_data[4])
        # (B, seq_len, D)
        batch_embs = batch_embs['last_hidden_state']
        batch_embs = self.drop(batch_embs)
        logits, pred = self.zero_shot_match(
            batch_embs, choice_idx, batch_data[1], batch_data[2])
        return logits, pred
