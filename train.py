import argparse

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def train(dir_model=None):
    # build model
    config = Config()
    if dir_model is not None:
        config.dir_model = dir_model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='filename of output weights', default=None)
    args = parser.parse_args()
    train(dir_model=args.output)
