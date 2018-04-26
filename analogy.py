from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import numpy as np

UNK = "$UNK$"
NUM = "$NUM$"


def is_num(word):
    try:
        word_num = list(word)
        while "," in word_num:
            word_num.remove(",")
        float("".join(word_num))
        return True
    except ValueError:
        pass
    return False


def get_embedding():
    config = Config()
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)

    embeddings = model.get_embeddings(test)
    dic = dict()
    for _word in config.vocab_words.keys():
        if _word != UNK and _word != NUM:
            word = _word.lower() if config.lowercase else _word
            word = NUM if is_num(word) else word
        else:
            word = _word
        dic[word] = embeddings[config.vocab_words[word]]
    return dic


def analogy():
    config = Config()
    dic = get_embedding()
    vocab = dic.keys()
    sum_questions = 0
    sum_right = 0
    with open(config.filename_questions, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            s = line.strip().split()
            if len(s) == 4:
                if config.lowercase:
                    for i in range(4):
                        s[i] = s[i].lower() if config.lowercase else s[i]
                        s[i] = NUM if is_num(s[i]) else s[i]
                        s[i] = s[i].lower()
                if s[0] in vocab and s[1] in vocab and s[2] in vocab and s[3] in vocab:
                    sum_questions += 1
                    x = dic[s[2]] + dic[s[1]] - dic[s[0]]
                    ans = None
                    for i in dic:
                        def get_distance(x, y):
                            return np.sum((x - y) ** 2)

                        if ans is None or get_distance(dic[i], x) < get_distance(dic[ans], x):
                            ans = i
                    if ans == s[3]:
                        sum_right += 1

    print(sum_right / sum_questions * 100.0)


if __name__ == "__main__":
    analogy()
