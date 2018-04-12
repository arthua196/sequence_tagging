import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='filename of output weights', default=None)
    parser.add_argument('--kth_fold', type=int, help='time of k fold', default=None)
    parser.add_argument('--action', type=str, help='evaluate or train&evaluate', choices=['evaluate', 'train'],
                        default='train')
    args = parser.parse_args()

    if args.kth_fold is not None:
        os.system("python3 build_data.py --kth_fold " + str(args.kth_fold))
    else:
        os.system("python3 build_data.py")

    if args.output is not None:
        if args.action == 'train':
            os.system("python3 train.py --output " + args.output)
        os.system("python3 evaluate.py --output " + args.output)
    else:
        if args.action == 'train':
            os.system("python3 train.py")
        os.system("python3 evaluate.py")
