import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy

def data_iterator(device, batch_size=128):
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings (tokens) 
        """
        return [tok.text for tok in spacy_de.tokenizer(text)]#[::-1]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize = tokenize_de, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    TRG = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                        fields = (SRC, TRG))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    print(vars(train_data.examples[0]))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    """
    Iterator help to get a batch of examples and pad to same (max) length.
    Then BucketIterator boost up the performance by minimizes the amount of padding 
    in both the source and target sentences
    """
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size, 
        device = device)

    return train_iterator, valid_iterator, test_iterator, SRC, TRG, test_data


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator, _, _, test_data = data_iterator(device, batch_size=4)

    for i, batch in enumerate(train_iterator):
        src = batch.src
        trg = batch.trg

        if i <= 0:
            print(src.shape) # (seq, bs)
            print(trg.shape)

"""
Number of training examples: 29000
Number of validation examples: 1014
Number of testing examples: 1000
{'src': ['zwei', 'junge', 'weiße', 'männer', 'sind', 'im', 'freien', 'in', 'der', 'nähe', 'vieler', 'büsche', '.'], 
'trg': ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']}
Unique tokens in source (de) vocabulary: 7853
Unique tokens in target (en) vocabulary: 5893
tensor([[   2,    2,    2,    2],
        [   4,    4,    4,    4],
        [ 118, 1934,   75,  115],
        [  20,    0,   22,   17],
        [3319,   14, 5449,   85],
        [   6,   12,   17,  438],
        [  21,   48,   12,  302],
        [5343,  260,  279,    9],
        [1511, 1558,   73, 5044],
        [  18,  848,   76,   11],
        [   3,   11,    3,   10],
        [   1,   13,    1, 4151],
        [   1,    5,    1,    7],
        [   1,    3,    1,  470],
        [   1,    1,    1,    9],
        [   1,    1,    1,   16],
        [   1,    1,    1, 1341],
        [   1,    1,    1,    8],
        [   1,    1,    1,   10],
        [   1,    1,    1,  174],
        [   1,    1,    1,    8],
        [   1,    1,    1,    3]], device='cuda:0')
tensor([[   2,    2,    2,    2],
        [  16,    4,  113,   21],
        [1811,    9,   19,  115],
        [  17,   13,   17,   11],
        [3241,  163,  254,    4],
        [   8,  867,   54,  530],
        [   4,   97,  121,   14],
        [2820, 2142,   18,   15],
        [   6,  734, 2923,  405],
        [   7,  226,    5,   13],
        [  96,    4,    3,  213],
        [   5,  138,    1,  453],
        [   3,  281,    1,   11],
        [   1,    5,    1, 5474],
        [   1,    3,    1,   15],
        [   1,    1,    1,  527],
        [   1,    1,    1,  156],
        [   1,    1,    1,   18],
        [   1,    1,    1,  156],
        [   1,    1,    1,   54],
        [   1,    1,    1,    7],
        [   1,    1,    1,  116],
        [   1,    1,    1,    5],
        [   1,    1,    1,    3]], device='cuda:0')
"""



