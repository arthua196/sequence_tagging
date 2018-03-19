from model.config import Config
from train import train
from evaluate import evaluate
from build_data import build_data
from model.ner_model import NERModel
import argparse

if __name__ == "__main__":
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='filename of output weights', default=None)
    parser.add_argument('--kth_fold', type=int, help='time of k fold', default=None)
    args = parser.parse_args()

    if args.output is not None:
        config.dir_model = args.output
    if args.kth_fold is not None:
        build_data(kth_fold=args.kth_fold)
    else:
        build_data()

    model = NERModel(config)
    train(model=model)
    evaluate(model=model)
