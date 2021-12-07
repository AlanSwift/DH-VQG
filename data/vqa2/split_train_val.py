import random, pickle

def select(whole, keys):
    ret = {k: whole[k] for k in keys}
    return  ret

def split(train_total_path, train_output_path, val_output_path, ratio=0.1):
    with open(train_total_path, "rb") as f:
        whole_split = pickle.load(f)
    keys = list(whole_split.keys())
    random.shuffle(keys)
    train_num = int(len(keys) * (1 - ratio))
    train_keys = keys[:train_num]
    val_keys = keys[train_num:]

    train_split = select(whole_split, train_keys)
    val_split = select(whole_split, val_keys)

    with open(train_output_path, "wb") as f:
        pickle.dump(train_split, f)

    with open(val_output_path, "wb") as f:
        pickle.dump(val_split, f)




if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    # split(train_total_path="/home/shiina/data/aaai/vqa2/train_split_dic_unique.pkl",
    #       train_output_path="/home/shiina/data/aaai/vqa2/train_split_dic_unique_90.pkl",
    #       val_output_path="/home/shiina/data/aaai/vqa2/train_split_dic_unique_10.pkl",
    #       ratio=0.1)
    split(train_total_path="/home/shiina/data/aaai/vqa2/train_split_dic_unique_full_3.pkl",
          train_output_path="/home/shiina/data/aaai/vqa2/train_split_dic_unique_full_3_90.pkl",
          val_output_path="/home/shiina/data/aaai/vqa2/train_split_dic_unique_full_3_10.pkl",
          ratio=0.1)
