from os import path


class Dict(object):
    def __init__(self):
        self.w2idx = dict()
        self.idx2w = []

    def add_word(self, word):
        if word not in self.w2idx:
            self.w2idx[word] = len(self.idx2w)
            self.idx2w.append(word)

    def __len__(self):
        return len(self.idx2w)


class Corpus(object):
    def __init__(self, data_dir):
        self.dictionary = Dict()
        self.train_data = self.tokenize(path.join(data_dir, 'train_small.txt'))
        self.val_data = self.tokenize(path.join(data_dir, 'valid_small.txt'))
        self.test_data = self.tokenize(path.join(data_dir, 'test_small.txt'))

    def tokenize(self, file_path):
        assert path.exists(file_path)
        count = 0

        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.strip() == '' or line.strip()[0] == '=':
                    continue
                tokens = line.strip().split()
                count += len(tokens)
                for token in tokens:
                    self.dictionary.add_word(token)

        target = [0] * count
        count = 0
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.strip() == '' or line.strip()[0] == '=':
                    continue
                tokens = line.strip().split()
                for token in tokens:
                    target[count] = self.dictionary.w2idx[token]
                    count += 1
        return target


def get_data(opt):
    return Corpus(opt.data_dir)
