from model.config import Config
from train import train
from evaluate import evaluate
from build_data import build_data

if __name__ == "__main__":
    config = Config()
    if config.use_k_fold:
        for i in range(config.k_fold):
            build_data(kth_fold=i)
            config.dir_output = config.dir_output[:-1] + str(i) + "/"
            train(config)
            evaluate(config)
    else:
        train()
        evaluate()
