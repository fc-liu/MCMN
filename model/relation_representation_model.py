import torch
import torch.nn as nn
from torch import nn as nn

from configure import FLAGS
# from utils import position2mask

gpu_aval = torch.cuda.is_available()


# def extract_entity_2(x, e):
#     max_seq_len = x.shape[1]
#     e1 = e[:, 0]
#     mask = position2mask(e1, max_seq_len)[:, :, None]
#     try:
#         e_hiddens = torch.sum(x * mask, 1)  # (batch,hidden)
#     except Exception as e:
#         print(e)
#     return e_hiddens

def extract_entity(x, e):
    bsz=x.shape[0]
    emb=x[torch.arange(bsz),e[:,0]]
    return emb

class EntityMarkerEncoder(nn.Module):
    def __init__(self):
        super(EntityMarkerEncoder, self).__init__()
        self.active = torch.tanh
        self.layerNorm = nn.LayerNorm(FLAGS.hidden_size)
        self.output_sizes = (FLAGS.hidden_size, FLAGS.hidden_size)

    def forward(self, token_embs, pos1, pos2, mask):

        hidden1 = extract_entity(token_embs, pos1)
        # hidden1_2=extract_entity_2(token_embs,pos1)
        # assert torch.equal(hidden1,hidden1_2)
        hidden2 = extract_entity(token_embs, pos2)
        # hidden2_2=extract_entity_2(token_embs,pos2)
        # assert torch.equal(hidden2,hidden2_2)

        # hidden1 = self.active(hidden1)
        # hidden1 = self.layerNorm(hidden1)
        # hidden2 = self.active(hidden2)
        # hidden2 = self.layerNorm(hidden2)
        return (hidden1, hidden2)


class EntityMarkerClsEncoder(nn.Module):
    def __init__(self):
        super(EntityMarkerClsEncoder, self).__init__()
        self.output_sizes = [FLAGS.hidden_size]

    def forward(self, token_embs, pos1, pos2, mask):

        return token_embs[:, 0]


first = True


class RRModel(nn.Module):
    """
    Relation Represemtation Model
    """

    def __init__(self, embedder, relation_encoder):
        super(RRModel, self).__init__()
        self.sentence_encoder = nn.DataParallel(
            embedder, device_ids=FLAGS.paral_cuda)
        self.rel_encoder = relation_encoder
        # self.output_size = self.rel_encoder.output_size
        relation_hidden_size = 0
        for i in self.rel_encoder.output_sizes:
            relation_hidden_size += i
        self.relation_hidden_size = relation_hidden_size
        self.output_size = self.relation_hidden_size

    def forward(self, tokens, pos1, pos2, mask):
        # tokens: batch_size*seq_len
        # pos1,pos2: batch_size*[start, end]
        # mask: batch_size*seq_len
        # bert
        # encoded_layers, _ = self.sentence_encoder(
        #     tokens, output_all_encoded_layers=False)
        layer = FLAGS.bert_layer
        global first
        if first:
            # print(
            #     "#################################layer : {}######### #################".format(layer))
            first = False
        # with torch.no_grad():
        encoded_layers = self.sentence_encoder(
            tokens, attention_mask=mask)
        # print("###################size:{}###################".format(
        #     encoded_layers[0]))
        atts = encoded_layers[-1][-1]
        encoded_layers = encoded_layers[0]  # [layer]
        # mask[:,0]=0
        # mask[:,-1]=0
        relation_embs = self.rel_encoder(
            encoded_layers, pos1, pos2, mask)
        shape = len(relation_embs)
        if shape > 1:
            parts = []
            for part in relation_embs:
                parts.append(part)
            rel_rep = torch.cat(parts, dim=-1)
        else:
            rel_rep = relation_embs
        # rel_rep=relation_embs[0]
        return rel_rep
