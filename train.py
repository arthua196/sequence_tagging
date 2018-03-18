from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def train(config=Config(), model=None):
    # build model
    if model is None:
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
    train()
