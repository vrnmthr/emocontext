from torch.utils.data import Dataset
from os import path
import tokenizer
from tqdm import tqdm


class EmoContextDataset(Dataset):
    """
    Accesses the preprocessed EmoContext dataset
    """

    def __init__(self, dir, mode):
        """
        :param dir: directory containing various splits
        :param mode: 'dev', 'train' or 'test'
        """
        super(EmoContextDataset, self).__init__()
        fname = "{}_processed.txt".format(mode)
        self.mode = mode

        # has temp label placeholder
        self.label_to_id = {"happy": 0, "sad": 1, "angry": 2, "others": 3, "label": None}
        self.id_to_label = {self.label_to_id[label]: label for label in self.label_to_id}

        self.data = []
        with open(path.join(dir, fname), "r") as data:
            for line in tqdm(data, desc='Loading {} set'.format(mode)):
                split = line.strip().split("\t")
                # remove index
                del split[0]

                # tokenize all sentences that have valid lengths
                if len(split) >= 4:
                    tokenized = [x for x in split]
                    # keeps track if any sentence is empty
                    empty = False
                    for i in range(3):
                        tokenized[i] = tokenizer.tokenize(split[i])
                        empty = empty or not tokenized[i]
                    if not empty:
                        tokenized[3] = self.label_to_id[tokenized[3]]
                        self.data.append(tokenized)

        # first row is header
        del self.data[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """
        Gets the ith item in the dataset as a dict: {'item':, 'label':}
        Item is a list of 3 tokenized sentences
        """
        sample = self.data[i]
        return {"item": sample[:3], "label": sample[3]}

    @staticmethod
    def collate_fn(samples):
        """
        :return: a minibatch made up of a list of samples
        """
        return {"item": [s['item'] for s in samples], "label": [s['label'] for s in samples]}
