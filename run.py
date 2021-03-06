import os
import time
from model.config import Config
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, help='evaluate or train&evaluate', choices=['evaluate', 'train'],
                        default='train')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    time_start = time.time()
    config = Config(load=False)
    logger = config.logger
    logger.info("Running Path:\t" + os.getcwd())
    if config.use_k_fold:
        dir_model = config.dir_model[:-1]
        for i in range(config.k_fold):
            os.system(
                "python3 main.py --output %s --kth_fold %s --action %s" % (
                dir_model + str(i) + "/", str(i), args.action))
        os.system("python3 get_result.py >> " + config.dir_evaluate_result)
    else:
        os.system("python3 main.py --action %s" % (args.action))
    time_end = time.time()
    logger.info("Running Time: %.2f" % (time_end - time_start))
