import os
from model.config import Config

if __name__ == "__main__":
    print("Running Path:\t" + os.getcwd())
    config = Config()
    if config.use_k_fold:
        dir_model = config.dir_model[:-1]
        for i in range(config.k_fold):
            os.system("python3 main.py --output %s --kth_fold %s" % (dir_model + str(i) + "/", str(i)))
    else:
        os.system("python3 main.py")


