import argparse

from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word, make_fold_data


def build_data(kth_fold=None):
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...
        :param kth_fold:

    """
    # make k_fold
    config = Config(load=False)
    if config.use_k_fold and kth_fold is not None:
        make_fold_data(config.dir_k_fold, config.k_fold, kth_fold, config.filename_train, config.filename_test)

    # get config and processing of words
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev = CoNLLDataset(config.filename_dev, processing_word)
    test = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)

    # Build the TrainSet word vocab
    train_set_words, _ = get_vocabs([train])
    write_vocab(train_set_words, config.filename_trainset_words)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    if config.use_pretrained:
        vocab_glove = get_glove_vocab(config.filename_glove)
    else:
        vocab_glove = vocab_words
        # vocab_glove = train_set_words
    write_vocab(vocab_glove, config.filename_embedding_words)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    if config.use_pretrained:
        export_trimmed_glove_vectors(vocab, config.filename_glove,
                                     config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kth_fold', type=int, help='time of k fold', default=None)
    args = parser.parse_args()
    build_data(args.kth_fold)
