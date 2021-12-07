import pickle
import numpy as np


def get_ratio(split_path):
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    cnt_0_all = 0
    cnt_1_all = 0
    cnt = 0
    for key, item in split.items():
        vh = item['visual_hint']
        vh = np.array(vh)
        cnt_0 = (vh == 0).sum()
        cnt_1 = (vh == 1).sum()
        cnt_0_all += cnt_0
        cnt_1_all += cnt_1
        cnt += 1
    print(cnt_0_all / cnt)
    print(cnt_1_all / cnt)
    pass


if __name__ == "__main__":
    get_ratio(split_path="/home/shiina/data/aaai/vqa2/val_split_dic_unique.pkl")
    pass