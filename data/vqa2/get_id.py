import os, argparse, pickle, json

def get_id(items):
    ids = list(items.keys())
    id_dict = {id: 1 for id in ids}
    return id_dict
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_split', default="/home/shiina/data/aaai/vqa2/train_split_dic_unique_90.pkl", type=str,
                        help='path to the config file')
    parser.add_argument('--val_split', default="/home/shiina/data/aaai/vqa2/val_split_dic_unique.pkl", type=str, help='rank')
    parser.add_argument('--train_ids_output', type=str,
                        default="/home/shiina/shiina/question/iQAN/data/vqa2/our_ids/train_ids_90.pkl", help="")
    parser.add_argument('--val_ids_output', type=str,
                        default="/home/shiina/shiina/question/iQAN/data/vqa2/our_ids/val_ids.pkl", help="")

    parser.add_argument('--train_ids_json', type=str,
                        default="/home/shiina/shiina/question/iq/data/vqa/train_ids_90.json", help="")
    parser.add_argument('--val_ids_json', type=str,
                        default="/home/shiina/shiina/question/iq/data/vqa/val_ids.json", help="")

    args = parser.parse_args()
    with open(args.train_split, "rb") as f:
        train_split = pickle.load(f)
    with open(args.val_split, "rb") as f:
        val_split = pickle.load(f)

    train_ids = get_id(train_split)
    val_ids = get_id(val_split)
    # with open(args.train_ids_output, "wb") as f:
    #     pickle.dump(train_ids, f)
    # with open(args.val_ids_output, "wb") as f:
    #     pickle.dump(val_ids, f)

    with open(args.train_ids_json, "w") as f:
        json.dump(train_ids, f)
    with open(args.val_ids_json, "w") as f:
        json.dump(val_ids, f)
