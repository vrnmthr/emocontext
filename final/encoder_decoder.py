"""
General approach:
- restrict single sentences to 15 in length
- pad every total conversation with <pad> token, but keep track of where each sentence begins/ends
- when encoding, encode a full batch at a time using packed-sequence encoding
- when decoding, decode a full batch *one word at a time* and calculate the loss for the whole thi
"""

import os
import argparse
import logging
import multiprocessing

import numpy as np

# Torch Imports
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm
from elmo import Elmo
from cornell_dataset import CornellMovieDialogs
from reddit_dataset import RedditDataset

LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 32

LOSS_FUNC = nn.CosineEmbeddingLoss()
OPTIMIZER = optim.Adam

EOS = "|"


class Encoder(nn.Module):
    """
    GRU-encoder that encodes a batched list of sentences
    """
    def __init__(self, batch_size, device=torch.device('cpu')):
        super(Encoder, self).__init__()
        self.hidden_size = 2048
        self.input_size = 1024
        self.batch_size = batch_size
        self.device = device

        self.elmo = Elmo(device)
        # TODO: do we want to add layers and/or dropout?
        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.to(device)

    def init_hidden(self):
        """
        Returns a new value for the hidden layer of the GRU cell
        """
        return torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)

    def forward(self, tokens, seq_lens):
        """
        :param tokens: list of sentences (batch, seq_len), do NOT need to be padded
        :param seq_lens: original sequence lengths of each input (prior to padding) (batch_size)
        :returns: tensor of (batch_size, hidden) representing hidden state for each sentence
        """

        # 1. Get embeddings of dimension (batch, seq_len, 1024)
        # ELMo pads everything for us which is lovely
        embedded = self.elmo(tokens)

        # 2. Sort seq_lens and embeds in descending order of seq_lens
        sorted_seq_lens, perm_ix = torch.sort(seq_lens, descending=True)
        sorted_embeddings = embedded[perm_ix]

        # 3. Obtain a PackedSequence object from pack_padded_sequence.
        #    Be sure to pass batch_first=True as the first dimension of our input is the batch dim.
        packed_seq = pack_padded_sequence(sorted_embeddings, sorted_seq_lens, batch_first=True)

        # 4. Apply the RNN over the sequence of packed embeddings to obtain a sentence encoding.
        #    Reset the hidden state each time
        hidden = self.init_hidden()

        # encoding is (batch_size, seq_len, hidden_sz), hidden is (num_layers * num_directions, batch_size, hidden_sz)
        encoding, hidden = self.gru(packed_seq, hidden)
        # Extract the hidden state from the top layer of the GRU (batch, hidden)
        out = hidden[-1]

        # 6. Remember to unsort the output from step 5. If you sorted seq_lens and obtained a permutation
        #    over its indices (perm_ix), then the sorted indices over perm_ix will "unsort".
        #    For example:
        #       _, unperm_ix = perm_ix.sort(0)
        #       output = x[unperm_ix]
        #       return output
        _, unperm_ix = torch.sort(perm_ix)
        out = out[unperm_ix]
        return out


class Decoder(nn.Module):
    """
    Decoder that will decode a batch of sentences by one timestep
    """
    def __init__(self, device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.hidden_size = 2048
        self.input_size = 1024
        self.output_size = 1024
        self.device = device

        self.embed = nn.Linear(self.input_size, self.input_size)
        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.to(device)

    def forward(self, prev, hidden):
        """
        Runs one batch through the decoder
        :param prev: previous words predicted for the batch (batch, 1, 1024)
        :param hidden: previous hidden states for the batch (layers*direction, batch, hidden)
        :return: (batch_size, output_size) prediction for each batch
        """
        # compute some function of the previous ELMo embeddings
        output = self.embed(prev)
        output = F.relu(output)

        # run this one-step sequence through the GRU
        # output is (batch, 1, num_directions * hidden_size)
        output, hidden = self.gru(output, hidden)

        # project the final output into ELMo space
        # TODO: do we want a non-linear activation function at the end here? If so, which?
        output = self.out(output)
        return torch.squeeze(output, dim=1), hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(self.device)


def train_batch(seqs, seq_lens, targets, target_lens, encoder, decoder, optims):
    """
    Trains the encoder/decoder module on a single sentence batch
    :param seqs: numpy array of (batch, seq_len), containing padded sequences with EOS token
    :param seq_lens: torch tensor (batch) containing length of sequence pre-padding
    :param targets: torch tensor (batch, seq_len, 1024) of embedded, padded targets
    :param target_lens: numpy array (batch) of length of targets pre-padding
    :param optims: dict taking models to their respective optimizers
    :returns: average loss over the entire third sentence over the entire batch
    """

    # reset gradients for all modules
    optims[encoder].zero_grad()
    optims[decoder].zero_grad()

    loss = run_batch(seqs, seq_lens, targets, target_lens, encoder, decoder)

    loss.backward()
    optims[encoder].step()
    optims[decoder].step()
    return loss.item()


def run_batch(seqs, seq_lens, targets, target_lens, encoder, decoder):
    """
    runs the encoder/decoder module on a single sentence batch
    :param seqs: numpy array of (batch, seq_len), containing padded sequences with EOS token
    :param seq_lens: torch tensor (batch) containing length of sequence pre-padding
    :param targets: torch tensor (batch, seq_len, 1024) of embedded, padded targets
    :param target_lens: numpy array (batch) of length of targets pre-padding
    :param optims: dict taking models to their respective optimizers
    :returns: average loss over the entire third sentence over the entire batch
    """
    loss = 0

    # calculate encoding for the entire batch (batch, hidden)
    encoder_hidden = encoder(seqs, seq_lens)

    # predict words one by one
    # convert targets into (seq_len, batch, 1024)
    targets = torch.transpose(targets, 0, 1)
    decoder_hidden = torch.unsqueeze(encoder_hidden, 0)
    # TODO: come up with some smarter way to embed a <SOS> token
    decoder_input = torch.zeros((BATCH_SIZE, 1, 1024), dtype=torch.float, device=DEV)

    for i in range(torch.max(target_lens).item()):
        target = targets[i]
        # target (batch, 1024) is all ith targets for words across the batch
        output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_input = torch.unsqueeze(target, dim=1)
        # only compute loss for appropriate sentences
        label = torch.Tensor([1 if i < target_lens[j] else 0 for j in range(BATCH_SIZE)]).to(DEV)
        # computes average loss over the batch
        loss += LOSS_FUNC(output, target, label)

    return loss


def train():
    modes = ['train', 'dev']

    datasets = {mode: RedditDataset(args.data, mode) for mode in modes}
    data_sizes = {mode: len(datasets[mode]) for mode in modes}
    dataloaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            collate_fn=datasets[mode].collate_fn) for mode in modes
    }

    # TODO: should we be using the same ELMo as in the encoder in order to match the embeddings?
    elmo = Elmo(device=DEV)
    elmo.eval()
    encoder = Encoder(BATCH_SIZE, device=DEV)
    decoder = Decoder(device=DEV)
    print(encoder)
    print(decoder)
    encoder_optimizer = OPTIMIZER(encoder.parameters(), LEARNING_RATE)
    decoder_optimizer = OPTIMIZER(decoder.parameters(), LEARNING_RATE)
    optims = {encoder: encoder_optimizer, decoder: decoder_optimizer}

    train_loss = []  # training loss per epoch, averaged over batches
    dev_loss = []  # dev loss each epoch, averaged over batches

    # can map mode -> list to append to the appropriate list
    losses = {'train': train_loss, 'dev': dev_loss}

    print("starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        for mode in modes:
            running_loss = 0.0
            for batch in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, NUM_EPOCHS)):
                # get context vector
                context = [triple[0] + [EOS] + triple[1] for triple in batch]
                seq_lens = torch.Tensor([len(s) for s in context]).int().to(DEV)
                targets = [triple[2] for triple in batch]
                target_lens = torch.Tensor([len(s) for s in targets]).int().to(DEV)
                # ELMo will pad everything automatically
                with torch.no_grad():
                    targets = elmo(targets)

                if mode == 'train':
                    # ensure models are getting trained
                    encoder.train()
                    decoder.train()
                    loss = train_batch(context, seq_lens, targets, target_lens, encoder, decoder, optims)
                    running_loss += loss
                else:
                    encoder.eval()
                    decoder.eval()
                    with torch.no_grad():
                        loss = run_batch(context, seq_lens, targets, target_lens, encoder, decoder)
                        running_loss += loss

            avg_loss = running_loss / (data_sizes[mode] / BATCH_SIZE)
            losses[mode].append(avg_loss)
            print("{} Loss: {}".format(mode, avg_loss))

        # if the development set has more loss than it did last time, break
        # if len(dev_loss) > 2 and dev_loss[-1] > dev_loss[-2]:
        #     print("dev set loss increased at epoch {}".format(epoch))
        #     break

    return encoder, decoder, train_loss, dev_loss


def main():
    """
    Either load and evaluate a trained model, or train and save a model
    """

    if args.restore is not None:

        # find appropriate folder and load model
        print("loading model...")

    else:

        # actually train the model
        encoder, decoder, train_loss, dev_loss = train()

        # get the folder to save the output
        run_id = 0
        while os.path.isdir("saved_runs/encoder_decoder/run{}".format(run_id)):
            run_id += 1
        folder = "saved_runs/encoder_decoder/run{}".format(run_id)
        print("making output folder {}...".format(folder))
        os.mkdir(folder)

        # save training parameters
        params = "{}\n\n{}\n{}\nlearning-rate: {}\nbatch-size: {}\nepochs: {}\n\noptimizer: {}\nloss: {}\n".format(
            args.message, encoder, decoder, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, str(OPTIMIZER), str(LOSS_FUNC)
        )
        with open(os.path.join(folder, "params.txt"), "w") as fparams:
            fparams.write(params)

        # save weights
        torch.save(encoder.state_dict(), os.path.join(folder, "encoder_weights.pt"))
        torch.save(decoder.state_dict(), os.path.join(folder, "decoder_weights.pt"))

        # save losses
        np.save(os.path.join(folder, "train_loss.npy"), np.array(train_loss))
        np.save(os.path.join(folder, "dev_loss.npy"), np.array(dev_loss))

        # graph losses
        # f, axarr = plt.subplots(2, sharex=True)
        # axarr[0].plot(train_loss)
        # axarr[0].set_title('L: train_loss; R: dev_loss')
        # axarr[1].plot(dev_loss)
        # plt.savefig(os.path.join(folder, "loss.png"))


if __name__ == '__main__':
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="path to data directory")
    parser.add_argument("--device", type=str, help="cuda for gpu and cpu otherwise", default="cpu")
    parser.add_argument("--restore", type=int, help="run id of model to restore", default=None)
    parser.add_argument("--message", type=str, help="message describing this run", default="")
    args = parser.parse_args()

    DEV = torch.device(args.device)

    main()
