import torch
import torch.nn as nn
import torch.nn.functional as f
from prettytable import PrettyTable

from nqa.inputters import PAD
from nqa.modules.embeddings import Embeddings
from nqa.encoders.rnn_encoder import RNNEncoder
from nqa.modules.highway import Highway
from nqa.modules.char_embedding import CharEmbedding
from nqa.inputters import constants
from nqa.modules.matrix_attn import MatrixAttention


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()

        self.input_size = 0

        # one of them must be true
        assert args.use_word or args.use_chars

        self.use_word = args.use_word
        if self.use_word:
            self.word_embeddings = Embeddings(args.emsize,
                                              args.vocab_size,
                                              PAD)
            self.input_size += args.emsize

        self.use_chars = args.use_chars
        if self.use_chars:
            assert len(args.filter_size) == len(args.nfilters)
            self.char_embeddings = CharEmbedding(args.n_characters,
                                                 args.char_emsize,
                                                 args.filter_size,
                                                 args.nfilters)
            self.input_size += sum(list(map(int, args.nfilters)))
            self.highway_net = Highway(self.input_size, num_layers=2)

        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(self,
                sequence,
                sequence_char=None):

        word_rep = None
        if self.use_word:
            word_rep = self.word_embeddings(sequence.unsqueeze(2))  # B x P x d

        if self.use_chars:
            char_rep = self.char_embeddings(sequence_char)  # B x P x f
            word_rep = char_rep if word_rep is None \
                else torch.cat((word_rep, char_rep), 2)  # B x P x d+f
            word_rep = self.highway_net(word_rep)  # B x P x d+f

        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()
        self.encoder = RNNEncoder(args.rnn_type,
                                  input_size + args.extra_feat,  # NOTE here
                                  args.bidirection,
                                  args.nlayers,
                                  args.nhid,
                                  args.dropout_rnn,
                                  use_last=False)

        self.hidden_size = args.nhid
        self.dropout = nn.Dropout(p=args.dropout_rnn)

    def forward(self, input, input_len):
        # M: batch_size x seq_len x nhid*nlayers
        hidden, M = self.encoder(input, input_len)
        # M: batch_size x seq_len x nhid
        M = M.split(self.hidden_size, dim=2)[-1]
        M = self.dropout(M)
        return hidden, M


# model details can be found at https://arxiv.org/pdf/1611.01603.pdf
class BIDAF(nn.Module):
    """Bidirectional Attention Flow Network that finds answer span for the question from the given passage."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(BIDAF, self).__init__()

        self.embedder = Embedder(args)
        self.ctx_embd_layer = Encoder(args, self.embedder.input_size)

        self.matrix_attn_layer = MatrixAttention(args.nhid)

        self.modeling_layer = Encoder(args, self.ctx_embd_layer.hidden_size * 4)
        self.dropout = nn.Dropout(args.dropout)

        input_size = self.ctx_embd_layer.hidden_size * 5
        self.classifier = nn.Linear(input_size, 2, bias=False)

    def forward(self,
                sentence_word_rep,
                sentence_char_rep,
                sentence_len,
                question_word_rep,
                question_char_rep,
                question_len):
        """
        Input:
            - sentence_word_rep: ``(batch_size, max_sent_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        X = self.embedder(sentence_word_rep, sentence_char_rep)
        Q = self.embedder(question_word_rep, question_char_rep)

        # ------------- Contextual Embedding Layer -------------
        tensor_H = self.ctx_embd_layer(X, sentence_len)[1]  # B x P x h
        tensor_U = self.ctx_embd_layer(Q, question_len)[1]  # B x Q x h

        # ------------- Attention Flow Layer -------------
        # compute \alpha(h,u)
        S = self.matrix_attn_layer(tensor_H, tensor_U)  # B x P x Q

        # c2q: context-to-query attention
        c2q = torch.bmm(f.softmax(S, dim=2), tensor_U)  # B x P x h

        # q2c: query-to-context attention
        b = f.softmax(torch.max(S, 2)[0], dim=-1)  # B x P
        q2c = torch.bmm(b.unsqueeze(1), tensor_H)  # B x 1 x h
        q2c = q2c.repeat(1, tensor_H.size(1), 1)  # B x P x h , tiled P times

        # G: query aware representation of each context word
        G = torch.cat((tensor_H, c2q, tensor_H.mul(c2q), tensor_H.mul(q2c)), 2)  # B x P x 4h

        # ------------- Modeling Layer -------------
        hidden, M = self.modeling_layer(G, None)
        G_M = torch.cat((G, M), 2)  # B x P x 5h/(l + h)
        G_M = self.pool(G_M, sentence_word_rep.eq(constants.PAD).unsqueeze(2))  # B x 5h/(l + h)
        score = self.classifier(G_M)  # B x 2

        return score

    @staticmethod
    def pool(h, mask, type='max'):
        if type == 'max':
            h = h.masked_fill(mask, -constants.INFINITY_NUMBER)
            return torch.max(h, 1)[0]
        elif type == 'avg':
            h = h.masked_fill(mask, 0)
            return h.sum(1) / (mask.size(1) - mask.float().sum(1))
        else:
            h = h.masked_fill(mask, 0)
            return h.sum(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
