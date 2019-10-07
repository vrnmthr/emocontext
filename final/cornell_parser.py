"""
Writes dialogs from the cornell movie-dialog corpus into a single file
"""

import argparse
from os import path
import ast
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, help="path to cornell movie dialogs directory", default="data/cornell_corpus")
args = parser.parse_args()

conversations = open(path.join(args.dir, "movie_conversations.txt"), "r")
lines = open(path.join(args.dir, "movie_new.txt"), "r")
output = open(path.join(args.dir, "parsed_conversations.txt"), "w+")

# reads every line from lines into a map
splitter = " +++$+++ "
id_to_line = {}
print("Reading lines...")
for line in tqdm(lines):
    s = line.strip().split(splitter)
    if len(s) > 4:
        id = s[0].strip()
        line = s[4].strip()
        # only want to keep short lines because long lines will confuse the model
        if len(line) < 150:
            id_to_line[id] = line

print("Parsing conversations...")
for c in tqdm(conversations):
    s = c.split(splitter)
    ids = ast.literal_eval(s[3].strip())
    if len(ids) > 2:
        for i in range(len(ids) - 3):
            # need to catch errors because some lines got skipped above
            if ids[i] in id_to_line and ids[i+1] in id_to_line and ids[i+2] in id_to_line:
                conversation = [id_to_line[id] for id in ids[i:i+3]]
                output.write(splitter.join(conversation) + "\n")


