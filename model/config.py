import os
import tensorflow as tf

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_trainset_word = load_vocab(self.filename_trainset_words)
        self.vocab_embedding_word = load_vocab(self.filename_embedding_words)
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags,
                                                  lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                           if self.use_pretrained else None)

    # general config
    dir_output = "results/"
    dir_model = dir_output + "model.weights"
    dir_2nd_model = dir_output + "model_2nd.weights"
    path_log = dir_output + "log.txt"
    dir_evaluate_result = dir_output + "evaluate_result.txt"

    # for output mistakes in evaluate
    write_mistake_2file = True
    filename_wrong_preds = dir_output + "wrong_preds.txt"

    lowercase = True
    # embeddings
    dim_word = 100
    dim_char = 40

    # glove files
    filename_glove = "data/emb/emb{}.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/emb{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_dev = "data/corpus/valid.txt"
    filename_test = "data/corpus/test.txt"
    filename_train = "data/corpus/train.txt"

    max_iter = None  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # for judge ooxv word
    filename_trainset_words = "data/trainset_words.txt"
    filename_embedding_words = "data/embedding_words.txt"

    # k fold
    use_k_fold = False
    k_fold = 4
    dir_k_fold = "data/k_fold/"

    # embedding setting
    train_embeddings = False
    copy_embeddings = False
    use_projection = False
    use_residual = False
    use_attention = False
    use_projection_regularizer = False
    embedding_projection_type = "linear"
    projection_w_initilization = "xavier"

    # training
    nepochs = 2
    # nepochs = 50
    dropout = 0.5
    batch_size = 16
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 5

    # model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU

    # GPU config
    gpuConfig = tf.ConfigProto()
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    gpuConfig.gpu_options.allow_growth = True
