import pandas as pd

res = pd.DataFrame()
with open("res_all.txt", "r", encoding="utf-8") as fin:
    for line in fin.readlines():
        s = line.split()
        lr = s[0]
        data = list(i for i in s[1:])
        for i in range(len(data)):
            data[i] = eval(data[i])
        x = pd.DataFrame(data=data, index=["acc", "f1", "iv_f1", "oobv_f1", "ooev_f1", "ootv_f1"], columns=[lr])
        res = pd.concat([res, x], axis=1)
res = res.sort_index(1)
print(res)
