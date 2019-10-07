"""
Defines a module that generates ELMo embeddings for input sentences
"""

import torch
from torch import nn

from allennlp.modules.elmo import Elmo as allen_elmo
from allennlp.modules.elmo import batch_to_ids


class Elmo(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        """ Load the ELMo model. The first time you run this, it will download a pretrained model. """
        super(Elmo, self).__init__()
        options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # initialize instance of Elmo model
        self.elmo = allen_elmo(options, weights, 2, dropout=0)
        self._dev = device
        self.to(device)

    def forward(self, batch):
        """
        :param batch: List of tokenized sentences of varying lengths
        :return: Embeddings of dimension (batch, seq_len, embedding_dim), where seq_len is the max-length
        """
        # embeddings['elmo_representations'] is length two list of tensors.
        # Each element contains one layer of ELMo representations with shape
        # (batch, seq_len, embedding_dim).
        char_ids = batch_to_ids(batch).to(self._dev)
        embeddings = self.elmo(char_ids)
        return embeddings['elmo_representations'][-1]
