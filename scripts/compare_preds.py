# compare two preds files
import sys

RIGHT = "[Right Predicts]"
WRONG = "[Wrong Predicts]"
NOT = "[Not Predicts]"
END = "[end]"

filename1 = sys.argv[1]
filename2 = sys.argv[2]


def read_sen(filename):
    with open(filename, "r", encoding="utf-8") as fin:
        sen = ""
        status = "sen"
        res = dict()
        res["right"] = list()
        res["wrong"] = list()
        res["not"] = list()
        for line in fin.readlines():
            if line.strip() == END:
                yield res
                sen = ""
                status = "sen"
                res = dict()
                res["right"] = list()
                res["wrong"] = list()
                res["not"] = list()
            elif line.strip() == RIGHT:
                status = "right"
            elif line.strip() == WRONG:
                status = "wrong"
            elif line.strip() == NOT:
                status = "not"
            else:
                if status == "sen":
                    res["sen"] = line.strip().split(" ")
                elif status == "right":
                    s = line.strip().split("\t")
                    start_pos = eval(s[0])
                    end_pos = eval(s[1])
                    type = s[3]
                    res[status].append((start_pos, end_pos, type))
                else:
                    pass


if __name__ == "__main__":
    for (a, b) in zip(read_sen(filename1), read_sen(filename2)):
        if a["sen"] != b["sen"]:
            print("ERROR: The two sentences are not the same!")
            break
        else:
            s1 = set(a["right"])
            s2 = set(b["right"])
            sen = a["sen"]
            print(" ".join(sen))
            print("[%s - %s]" % (filename1, filename2))
            for chunk in s1 - s2:
                word = ""
                for i in range(chunk[0], chunk[1]):
                    word += sen[i]
                print(str(chunk[0]) + "\t" + str(chunk[1]) + "\t" + word + "\t" + chunk[2])
            print("[%s - %s]" % (filename2, filename1))
            for chunk in s2 - s1:
                word = ""
                for i in range(chunk[0], chunk[1]):
                    word += sen[i]
                print(str(chunk[0]) + "\t" + str(chunk[1]) + "\t" + word + "\t" + chunk[2])
            print("[end]")
