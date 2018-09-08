import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx2count = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.idx2count.append(1)
            self.word2idx[word] = len(self.idx2word) - 1
        else:
            self.idx2count[self.word2idx[word]] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path_train, path_test):
        self.dictionary_title = Dictionary()
        self.dictionary_authors = Dictionary()

        # training set is ICLR 2017 list.
        # you can add more training set as much as you want!
        self.add_corpus(os.path.join(path_train, 'ICLR_2017_accepted.txt'))
        self.add_corpus(os.path.join(path_train, 'ICLR_2017_rejected.txt'))

        # test set is ICLR 2018 list. DO NOT MODIFY!
        self.add_corpus(os.path.join(path_test, 'ICLR_2018_accepted.txt'))
        self.add_corpus(os.path.join(path_test, 'ICLR_2018_rejected.txt'))



        # sort the words by word frequency in descending order
        idx_argsorted_title = np.flip(np.argsort(self.dictionary_title.idx2count), axis=-1)
        idx_argsorted_authors = np.flip(np.argsort(self.dictionary_authors.idx2count), axis=-1)

        # re-create given the sorted ones
        self.dictionary_title.idx2count = np.array(self.dictionary_title.idx2count)[idx_argsorted_title].tolist()
        self.dictionary_title.idx2word = np.array(self.dictionary_title.idx2word)[idx_argsorted_title].tolist()
        self.dictionary_title.word2idx = dict(zip(self.dictionary_title.idx2word,
                                            np.arange(len(self.dictionary_title.idx2word)).tolist()))
        self.dictionary_authors.idx2count = np.array(self.dictionary_authors.idx2count)[idx_argsorted_authors].tolist()
        self.dictionary_authors.idx2word = np.array(self.dictionary_authors.idx2word)[idx_argsorted_authors].tolist()
        self.dictionary_authors.word2idx = dict(zip(self.dictionary_authors.idx2word,
                                            np.arange(len(self.dictionary_authors.idx2word)).tolist()))

        self.train_accepted = self.tokenize(os.path.join(path_train, 'ICLR_2017_accepted.txt'))
        self.train_rejected = self.tokenize(os.path.join(path_train, 'ICLR_2017_rejected.txt'))

        self.test_accepted = self.tokenize(os.path.join(path_test, 'ICLR_2018_accepted.txt'))
        self.test_rejected = self.tokenize(os.path.join(path_test, 'ICLR_2018_rejected.txt'))

    def add_corpus(self, path):
        """Tokenizes a txt file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            line_count = 0
            tokens_title = 0
            tokens_authors = 0
            # each row has: (paper name) + (tab) + (authors delimited by ", ")
            for row in f:
                # lowercase the string, split the title by space, and split the authors by ", "
                row = row.split('\t')
                title = row[0].lower().strip('\n').split()
                # ICLR 2017 authors has a dirty \xa0 instead of space. replace it
                authors = row[1].replace(u'\xa0', u' ')
                authors = authors.lower().strip('\n').split(', ')
                # increase the token count
                tokens_title += len(title)
                tokens_authors += len(authors)
                # add the word to the Dictionary
                for word in title:
                    self.dictionary_title.add_word(word)
                for word in authors:
                    self.dictionary_authors.add_word(word)

        #return tokens_title, tokens_authors

    def tokenize(self, path):
        ids_title = []
        ids_authors = []
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            for row in f:
                row = row.split('\t')
                title = row[0].lower().strip('\n').split()
                authors = row[1].replace(u'\xa0', u' ')
                authors = authors.lower().strip('\n').split(', ')

                id_title = []
                id_authors = []
                for word in title:
                    id_title.append(self.dictionary_title.word2idx[word])
                for word in authors:
                    id_authors.append(self.dictionary_authors.word2idx[word])
                ids_title.append(id_title)
                ids_authors.append(id_authors)

        return [ids_title, ids_authors]


class PaperDecisionDataset(Dataset):
    def __init__(self, x_title, x_authors, y):
        self.x_title = x_title
        self.x_authors = x_authors
        self.y = y
        assert len(self.x_title) == len(self.x_authors) == len(self.y)

    def __len__(self):
        return len(self.x_title)

    def __getitem__(self, index):
        return self.x_title[index], self.x_authors[index], self.y[index]


def create_corpus(path_train, path_test):
    corpus = Corpus(path_train, path_test)

    x_title_accepted, x_authors_accepted = corpus.train_accepted[0], corpus.train_accepted[1]
    # assign accepted papers as label zero
    y_accepted = np.zeros(len(corpus.train_accepted[0]), dtype=np.long).tolist()

    x_title_rejected, x_authors_rejected = corpus.train_rejected[0], corpus.train_rejected[1]
    # assign accepted papers as lable one
    y_rejected = np.ones(len(corpus.train_rejected[0]), dtype=np.long).tolist()

    x_title_train = x_title_accepted + x_title_rejected
    x_authors_train = x_authors_accepted + x_authors_rejected
    y_train = y_accepted + y_rejected


    x_title_accepted, x_authors_accepted = corpus.test_accepted[0], corpus.test_accepted[1]
    # assign accepted papers as label zero
    y_accepted = np.zeros(len(corpus.test_accepted[0]), dtype=np.long).tolist()

    x_title_rejected, x_authors_rejected = corpus.test_rejected[0], corpus.test_rejected[1]
    # assign accepted papers as lable one
    y_rejected = np.ones(len(corpus.test_rejected[0]), dtype=np.long).tolist()

    x_title_test = x_title_accepted + x_title_rejected
    x_authors_test = x_authors_accepted + x_authors_rejected
    y_test = y_accepted + y_rejected

    return corpus, x_title_train, x_authors_train, y_train, x_title_test, x_authors_test, y_test


def collate_fn(data):
    # custom collate fn for PaperDecisionDataset
    title = []
    authors = []
    decision = []
    for datapoint in data:
        title.append(datapoint[0])
        authors.append(datapoint[1])
        decision.append(datapoint[2])
    return title, authors, decision
