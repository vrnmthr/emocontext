import preprocessor as p
import nltk
import random


def tokenize(sentence: str):
    """
    Custom tokenizer that doesn't split on ASCII smileys while still retaining most of the NLTK tokenization power
    """

    # replace all smileys with key
    key = "TEMP{}".format(random.randint(100000, 999999))
    smileys = []
    tokens = sentence.split(" ")
    for i, token in enumerate(tokens):
        if p.is_smiley(token):
            smileys.append(token)
            tokens[i] = key

    # parse the whole sentence
    no_smileys = " ".join(tokens)
    tokens = nltk.word_tokenize(no_smileys)

    # replace all the smileys, in order
    j = 0
    for i, token in enumerate(tokens):
        if token == key:
            tokens[i] = smileys[j]
            j += 1

    return tokens
