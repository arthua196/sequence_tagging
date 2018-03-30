import argparse

from model.config import Config
import pandas as pd


def get_result(filename=None):
    config = Config(load=False)
    acc = list()
    f1 = list()
    iv_f1 = list()
    ootv_f1 = list()
    ooev_f1 = list()
    oobv_f1 = list()
    num = list()
    iv_num = list()
    ootv_num = list()
    ooev_num = list()
    oobv_num = list()
    filename = config.dir_evaluate_result if filename is None else filename
    with open(filename, "r", encoding="utf-8") as fin:
        line_num = 0
        dic = dict()
        for line in fin.readlines():
            s = line.split("-")
            for ss in s:
                dic[ss.split()[0]] = eval(ss.split()[1])
            if line_num % 2 == 1:
                acc.append(dic["acc"])
                f1.append(dic["f1"])
                iv_f1.append(dic["iv_f1"])
                ootv_f1.append(dic["ootv_f1"])
                oobv_f1.append(dic["oobv_f1"])
                ooev_f1.append(dic["ooev_f1"])
                num.append(dic["num"])
                iv_num.append(dic["iv_num"])
                ootv_num.append(dic["ootv_num"])
                ooev_num.append(dic["ooev_num"])
                oobv_num.append(dic["oobv_num"])
                dic = dict()
            line_num += 1
        df = pd.DataFrame({"acc": acc,
                           "f1": f1,
                           "iv_f1": iv_f1,
                           "ootv_f1": ootv_f1,
                           "ooev_f1": ooev_f1,
                           "oobv_f1": oobv_f1
                           # "num": num,
                           # "iv_num": iv_num,
                           # "ootv_num": ootv_num,
                           # "ooev_num": ooev_num,
                           # "oobv_num": oobv_num
                           })
        print(df)
        print("----------------------------------------\n")
        print(df.describe())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_evaluate_result', type=str, help='filename of evaluate result', default=None)
    args = parser.parse_args()
    get_result(args.dir_evaluate_result)
