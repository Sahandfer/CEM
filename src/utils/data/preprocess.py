import re
import nltk
import inflect
import contractions

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def expand_contractions(text):
    return " ".join([contractions.fix(word) for word in text.split()])


def remove_punc(text):
    return re.sub(r"[^\w\s]", "", text)


def remove_url(text):
    return re.sub(r"http\S+", "", text)


def remove_spaces(text):
    return re.sub(" +", " ", text).strip()


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def word_tokenize(text):
    return nltk.word_tokenize(text)