import re
import ftfy
import json
import copy
import spacy
import torch

import contextlib
import numpy as np

from tqdm import tqdm
from distutils.dir_util import mkpath


def load_existing_data_loader(data_loader, path):
    old_data_loader = torch.load(path)
    for attr in data_loader.__dict__.keys():
        if attr not in old_data_loader.__dict__.keys():
            continue
        setattr(data_loader, attr, getattr(old_data_loader, attr))


################################################################################
#
# Code Below taken from HuggingFace pytorch-openai-lm repository
#
################################################################################


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(
        r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""",
        r" \1 ",
        text,
    )
    text = re.sub(r"\s*\n\s*", " \n ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "tagger", "ner", "textcat", "lemmatizer"],
        )
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path, encoding="utf-8").read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [
                            self.encoder.get(t, 0)
                            for t in self.bpe(token.text.lower()).split(" ")
                        ]
                    )
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [
                            self.encoder.get(t, 0)
                            for t in self.bpe(token.text.lower()).split(" ")
                        ]
                    )
                texts_tokens.append(text_tokens)
        return texts_tokens


def make_new_tensor_from_list(items, device_num, dtype=torch.float32):
    if device_num is not None:
        device = torch.device("cuda:{}".format(device_num))
    else:
        device = torch.device("cpu")
    return torch.tensor(items, dtype=dtype, device=device)


def make_new_tensor_from_list(items, device_num, dtype=torch.float32):
    if device_num is not None:
        device = torch.device("cuda:{}".format(device_num))
    else:
        device = torch.device("cpu")
    return torch.tensor(items, dtype=dtype, device=device)


# is_dir look ast at whether the name we make
# should be a directory or a filename
def make_name(opt, prefix="", eval_=False, is_dir=True, set_epoch=None, do_epoch=True):
    string = prefix
    string += "{}-{}".format(opt.dataset, opt.exp)
    string += "/"
    string += "{}-{}-{}".format(opt.trainer, opt.cycle, opt.iters)
    string += "/"
    string += opt.model
    if opt.mle:
        string += "-{}".format(opt.mle)
    string += "/"
    string += make_name_string(opt.data) + "/"

    string += make_name_string(opt.net) + "/"
    string += make_name_string(opt.train.static) + "/"

    if eval_:
        string += make_name_string(opt.eval) + "/"
    # mkpath caches whether a directory has been created
    # In IPython, this can be a problem if the kernel is
    # not reset after a dir is deleted. Trying to recreate
    # that dir will be a problem because mkpath will think
    # the directory already exists
    if not is_dir:
        mkpath(string)
    string += make_name_string(opt.train.dynamic, True, do_epoch, set_epoch)
    if is_dir:
        mkpath(string)

    return string


def make_name_string(dict_, final=False, do_epoch=False, set_epoch=None):
    if final:
        if not do_epoch:
            string = "{}_{}_{}".format(dict_.lr, dict_.optim, dict_.bs)
        elif set_epoch is not None:
            string = "{}_{}_{}_{}".format(dict_.lr, dict_.optim, dict_.bs, set_epoch)
        else:
            string = "{}_{}_{}_{}".format(dict_.lr, dict_.optim, dict_.bs, dict_.epoch)

        return string

    string = ""

    for k, v in dict_.items():
        if type(v) == DD:
            continue
        if isinstance(v, list):
            val = "#".join(is_bool(str(vv)) for vv in v)
        else:
            val = is_bool(v)
        if string:
            string += "-"
        string += "{}_{}".format(k, val)

    return string


def is_bool(v):
    if str(v) == "False":
        return "F"
    elif str(v) == "True":
        return "T"
    return v


def generate_config_files(type_, key, name="base", eval_mode=False):
    with open("config/default.json".format(type_), "r") as f:
        base_config = json.load(f)
    with open("config/{}/default.json".format(type_), "r") as f:
        base_config_2 = json.load(f)
    if eval_mode:
        with open("config/{}/eval_changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)
    else:
        with open("config/{}/changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)

    base_config.update(base_config_2)

    if name in changes_by_machine:
        changes = changes_by_machine[name]
    else:
        changes = changes_by_machine["base"]

    # for param in changes[key]:
    #     base_config[param] = changes[key][param]

    replace_params(base_config, changes[key])

    mkpath("config/{}".format(type_))

    with open("config/{}/config_{}.json".format(type_, key), "w") as f:
        json.dump(base_config, f, indent=4)


def replace_params(base_config, changes):
    for param, value in changes.items():
        if isinstance(value, dict) and param in base_config:
            replace_params(base_config[param], changes[param])
        else:
            base_config[param] = value


def initialize_progress_bar(data_loader_list):
    num_examples = sum([len(tensor) for tensor in data_loader_list.values()])
    return set_progress_bar(num_examples)


def set_progress_bar(num_examples):
    bar = tqdm(total=num_examples)
    bar.update(0)
    return bar


def merge_list_of_dicts(L):
    result = {}
    for d in L:
        result.update(d)
    return result


def return_iterator_by_type(data_type):
    if isinstance(data_type, dict):
        iterator = data_type.items()
    else:
        iterator = enumerate(data_type)
    return iterator


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def flatten(outer):
    return [el for inner in outer for el in inner]


def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


# Taken from Jobman 0.1
class DD(dict):
    def __getattr__(self, attr):
        if attr == "__getstate__":
            return super(DD, self).__getstate__
        elif attr == "__setstate__":
            return super(DD, self).__setstate__
        elif attr == "__slots__":
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        # Safety check to ensure consistent behavior with __getattr__.
        assert attr not in ("__getstate__", "__setstate__", "__slots__")
        #         if attr.startswith('__'):
        #             return super(DD, self).__setattr__(attr, value)
        self[attr] = value

    def __str__(self):
        return "DD%s" % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.items():
            z[k] = copy.deepcopy(kv, memo)
        return z
