"""
Imports the Reddit Dataset from a given file
"""
from os import path
from torch.utils.data import Dataset
from tqdm import tqdm

# splits sentences
SPLITTER1 = "|"
# splits tokens
SPLITTER2 = " "


class RedditDataset(Dataset):

    def __init__(self, dir, mode):
        self.dialogs = []
        fpath = path.join(dir, "{}_truncated.txt".format(mode))
        with open(fpath) as f:
            for dialog in tqdm(f, desc="Loading Reddit {} dataset".format(mode)):
                lines = dialog.strip().split(SPLITTER1)
                self.dialogs.append([x.strip().split(SPLITTER2) for x in lines])

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, i):
        return self.dialogs[i]

    @staticmethod
    def collate_fn(samples):
        """
        :return: a minibatch made up of a list of samples
        """
        return samples
