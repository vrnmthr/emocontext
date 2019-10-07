"""
- remove all instances where the first two sentences combined are longer than 30 or last sentence > 15
- add EOS token in the middle of the sentence where appropriate
- save the length of sequence pre-padding
- pad the sequence with PAD token to get 31
- save the length of last sentence pre-padding
- pad last sentence to get 15
"""

from tqdm import tqdm

# splits sentences
SPLITTER1 = " +++$+++ "
# splits tokens
SPLITTER2 = " +*+ "

EOS = "<|>"
PAD = "<0>"

fpath = "../data/cornell_corpus/dev_tokenized.txt"
outpath = "../data/cornell_corpus/dev_processed.txt"

with open(fpath, "r") as data:
    with open(outpath, "w") as fout:
        for line in tqdm(data):
            sentences = line.strip().split(SPLITTER1)
            # PADDED_CONTEXT | LEN | PADDED_TARGET | LEN
            for i in range(3):
                sentences[i] = sentences[i].split(SPLITTER2)
            if len(sentences[0]) + len(sentences[1]) <= 30 and len(sentences[2]) <= 15:
                row = [None] * 4
                context = sentences[0] + [EOS] + sentences[1]
                row[0] = context
                row[1] = str(len(context))
                row[2] = sentences[2]
                row[3] = str(len(sentences[2]))
                # pad everything appropriately
                while len(context) < 31:
                    context.append(PAD)
                while len(sentences[2]) < 15:
                    sentences[2].append(PAD)

                row[0] = SPLITTER2.join(row[0])
                row[2] = SPLITTER2.join(row[2])
                fout.write(SPLITTER1.join(row) + "\n")
