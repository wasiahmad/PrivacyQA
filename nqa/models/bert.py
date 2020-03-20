import torch
import logging
import torch.nn as nn
from nqa.inputters import constants

from prettytable import PrettyTable
from nqa.transformers.modeling_bert import BertEmbeddings, \
    BertEncoder, BertPooler, BertLayerNorm

logger = logging.getLogger(__name__)


class BertQA(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, config):
        """"Constructor of the class."""
        super(BertQA, self).__init__()

        self.initializer_range = config.initializer_range

        config.output_attentions = True
        config.output_hidden_states = True

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_pretrained_weights(self, state_dict):
        def load_weights(layer):
            obj = layer
            for attr in attributes[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attributes[-1], torch.nn.Parameter(parameters))

        for name, parameters in state_dict.items():
            name = name.replace('gamma', 'weight')
            name = name.replace('beta', 'bias')
            name = name.split('.', 1)[1]
            if name.startswith('embeddings.'):
                attributes = name[11:].split('.')
                load_weights(self.embeddings)
            elif name.startswith('encoder.'):
                attributes = name[8:].split('.')
                load_weights(self.encoder)
            elif name.startswith('pooler.'):
                attributes = name[7:].split('.')
                load_weights(self.pooler)
            else:
                logger.warning('No weights loaded from %s' % name)

    def encode(self,
               source_ids,
               source_pos_ids,
               source_tok_type_ids,
               source_mask):

        if source_mask is None:
            source_mask = torch.ones_like(source_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        extended_source_mask = source_mask.unsqueeze(1).unsqueeze(2)
        extended_source_mask = extended_source_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_source_mask = (1.0 - extended_source_mask) * -10000.0

        src_embedding_output = self.embeddings(source_ids,
                                               position_ids=source_pos_ids,
                                               token_type_ids=source_tok_type_ids)
        encoder_outputs = self.encoder(src_embedding_output,
                                       extended_source_mask,
                                       head_mask=None)

        src_sequence_output = encoder_outputs[0]
        return src_sequence_output, extended_source_mask

    def forward(self,
                source_ids,
                source_pos_ids,
                source_type_ids,
                source_mask):

        src_sequence_output, extended_source_mask = self.encode(source_ids,
                                                                source_pos_ids,
                                                                source_type_ids,
                                                                source_mask)
        pooled_output = self.pooler(src_sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

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
