import os
import csv
from preprocessor import preprocess
from tokenizer import tokenize
from tqdm import tqdm


max_len = 25
with open("reddit_cleaned.csv", "r", encoding="utf-8") as fin:
    with open("reddit_processed.csv", "w", encoding="utf-8") as fout:
        reader = csv.reader(fin, delimiter=",")
        writer = csv.writer(fout, delimiter="|")
        bad = False

        for i, line in tqdm(enumerate(reader)):
            row = []
            for comment in line:
                if comment == "[deleted]" or comment == "[removed]":
                    bad = True
                    break

                if "|" in comment:
                    bad = True
                    break
                    
                tokenized = tokenize(preprocess(comment.strip()))
                if len(tokenized) > 25:
                    bad = True
                    break

                orig_len = len(tokenized)
                for i in range(0, 25 - len(tokenized)):
                    tokenized.append("<pad>")

                row.append(" ".join(tokenized))
                row.append(orig_len)
                        
            if len(row) == 6 and not bad:
                writer.writerow(row)

            bad = False
        
        print(max_len)

            