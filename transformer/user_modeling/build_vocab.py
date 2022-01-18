import nltk
import json
from tqdm.auto import tqdm
import pickle
import argparse
from collections import Counter


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def init_vocab(self):
        self.add_word("<pad>")
        self.add_word("<start>")
        self.add_word("<end>")
        self.add_word("<and>")
        self.add_word("<unk>")

    def save(self, file_name):
        data = {}
        data["word2idx"] = self.word2idx
        data["idx2word"] = self.idx2word
        data["idx"] = self.idx

        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)
        return

    def load(self, file_name):
        with open(file_name, "r") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]
        self.idx = data["idx"]
        return


def build_vocab(data, threshold):
    """Build a simple vocabulary wrapper."""
    print("total number of image pairs", len(data))
    counter = Counter()
    for i, captions in enumerate(tqdm(data)):
        # for caption in captions:
        tokens = nltk.tokenize.word_tokenize(captions.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.init_vocab()

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab


def load_data(files):
    data = []
    for f in files:
        with open(f) as fid:
            fdata = json.load(fid)
        for item in fdata:
            if not isinstance(item['captions'], list):
                item['captions'] = [item['captions']]
            data += item['captions']
    return data


def main(args):
    data = load_data(args.data_set_paths)
    vocab = build_vocab(data, threshold=args.threshold)
    vocab.save(args.save_output_path)
    print("Total vocabulary size: {}".format(len(vocab)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CAP_FILE = './data/captions/cap.*.json'
    # DICT_OUTPUT_FILE = './data/captions/{}.json'

    parser.add_argument("--data_set_paths", type=str, nargs="+")
    parser.add_argument("--save_output_path", type=str)
    parser.add_argument(
        "--threshold", type=int, default=2, help="minimum word count threshold"
    )
    args = parser.parse_args()
    main(args)
