#!/usr/bin/env python3

"""
Defines the no-context baseline.
This model:
- takes in ELMo embeddings for the third sentence
- concatenates them
- runs the final vector through a softmax classifier
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

from tqdm import tqdm
from elmo import Elmo
from emocontext_dataset import EmoContextDataset

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 8

LOSS = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam


class NoContext(nn.Module):
    """ NoContext sentiment classifier network """

    def __init__(self, device=torch.device('cpu')):
        """
        The constructor for our net. The architecture of this net is the following:

            ELMo -> Mean Pooling -> Linear Layer -> Linear Layer -> Softmax Layer

        :param hidden_sz (int): The size of our RNN's hidden state
        :param embedding_lookup (nn.Module): The module used to perform embedding lookups
        :param num_layers (int): The number of RNN cells we want in our net (c.f num_layers param in torch.nn.LSTMCell)
        """
        super(NoContext, self).__init__()
        self._dev = device

        self.elmo = Elmo(device)
        self.linear = nn.Sequential(
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4)
        )

        self.to(device)

    def forward(self, batch, seq_lens):
        """
        :param batch: list of tokenized sentences of length (batch_size)
        :param seq_lens: list of sequence lengths
        :returns: (batch_size x 2) logits for each class
        """
        # (batch, seq_len, embedding_dim)
        embeddings = self.elmo(batch)
        # average all items in sequence to get (batch, embedding_dim)
        multiplier = torch.diag(torch.Tensor([1/x for x in seq_lens])).to(self._dev)
        sum = torch.sum(embeddings, 1)
        avg = torch.mm(multiplier, sum)
        return self.linear(avg)


def train():
    """
    Trains the model, returning (model, train_loss, dev_loss)
    """
    modes = ['train', 'dev']

    datasets = {mode: EmoContextDataset(args.data, mode) for mode in modes}
    data_sizes = {mode: len(datasets[mode]) for mode in modes}
    dataloaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            collate_fn=datasets[mode].collate_fn) for mode in modes
    }

    model = NoContext(device=DEV)
    print(model)
    optimizer = OPTIMIZER(model.parameters(), LEARNING_RATE)

    train_loss = []  # training loss per epoch, averaged over batches
    dev_loss = []  # dev loss each epoch, averaged over batches

    # can map mode -> list to append to the appropriate list
    losses = {'train': train_loss, 'dev': dev_loss}

    print("starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        for mode in modes:
            running_loss = 0.0
            for batch in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, NUM_EPOCHS)):
                # only use the third sentence!!
                sentences = [c[2] for c in batch['item']]
                # print(sentences)
                seq_lens = [len(x) for x in sentences]
                # print(seq_lens)
                label = torch.LongTensor(batch['label']).to(DEV)
                # print(label)
                if mode == 'train':
                    model.train()  # tell pytorch to set the model to train mode
                    model.zero_grad()
                    logits = model(sentences, seq_lens)
                    loss = LOSS(logits, label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                else:
                    model.eval()
                    with torch.no_grad():
                        logits = model(sentences, seq_lens)
                        loss = LOSS(logits, label)
                        running_loss += loss.item()

            avg_loss = running_loss/(data_sizes[mode]/BATCH_SIZE)
            losses[mode].append(avg_loss)
            print("{} Loss: {}".format(mode, avg_loss))

        # if the development set has more loss than it did last time, break
        # if len(dev_loss) > 2 and dev_loss[-1] > dev_loss[-2]:
        #     print("dev set loss increased at epoch {}".format(epoch))
        #     break

    return model, train_loss, dev_loss


def evaluate(model) -> np.ndarray:
    """
    :return: The confusion matrix obtained by evaluating NoContext on the test data split
    """
    dataset = EmoContextDataset(args.data, 'test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=dataset.collate_fn)

    data_size = len(dataset)
    confusion = np.zeros((4, 4))
    for sample in tqdm(dataloader, desc='evaluating'):
        model.eval()
        with torch.no_grad():
            # only use the third sentence!!
            sentences = [c[2] for c in sample['item']]
            # print(sentences)
            seq_lens = [len(x) for x in sentences]
            # print(seq_lens)
            output = model(sentences, seq_lens)
            # print(output)
            tag = torch.argmax(output)
            # print(tag.item())
            # print(sample['label'])
            confusion[tag.item(), sample['label']] += 1

    confusion /= data_size
    confusion = np.round_(confusion, 3)
    return confusion


def main():
    """
    Either load and evaluate a trained model, or train and save a model
    """

    if args.restore is not None:

        # find appropriate folder and load model
        print("loading model...")
        folder = "saved_runs/no_context/run{}".format(args.restore)
        weights = os.path.join(folder, "weights.pt")
        model = NoContext(device=DEV)
        model.load_state_dict(torch.load(weights, map_location='cpu'))

        # calculate confusion matrix
        confusion = evaluate(model)
        accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)

        # write to file
        print("writing output...")
        with open(os.path.join(folder, "evaluation.txt"), "w") as fout:
            s = "accuracy: {}%\nconfusion matrix:\n{}".format(accuracy, confusion)
            fout.write(s)

    else:

        # actually train the model
        model, train_loss, dev_loss = train()

        # get the folder to save the output
        run_id = 0
        while os.path.isdir("saved_runs/no_context/run{}".format(run_id)):
            run_id += 1
        folder = "saved_runs/no_context/run{}".format(run_id)
        print("making output folder {}...".format(folder))
        os.mkdir(folder)

        # save training parameters
        params = "{}\n\n{}\n\nlearning-rate: {}\nbatch-size: {}\nepochs: {}\n\noptimizer: {}\nloss: {}\n".format(
            args.message, model, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, str(OPTIMIZER), str(LOSS)
        )
        with open(os.path.join(folder, "params.txt"), "w") as fparams:
            fparams.write(params)

        # save weights
        torch.save(model.state_dict(), os.path.join(folder, "weights.pt"))

        # save losses
        np.save(os.path.join(folder, "train_loss.npy"), np.array(train_loss))
        np.save(os.path.join(folder, "dev_loss.npy"), np.array(dev_loss))

        # graph losses
        # THIS SEEMS TO BE CAUSING PROBLEMS ON THE GRID SO WE WILL GRAPH AFTER
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
