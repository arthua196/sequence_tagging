from model.config import Config
from train import train
from evaluate import evaluate
from build_data import build_data
from model.ner_model import NERModel

if __name__ == "__main__":
    build_data()
    config = Config()
    model = NERModel(config)
    model.build()
    if config.use_k_fold:
        for i in range(config.k_fold):
            build_data(kth_fold=i)
            model.config.dir_model = model.config.dir_model[:-1] + str(i) + "/"
            train(config, model)
            evaluate(config, model)
    else:
        train(model=model)
        evaluate(model=model)
