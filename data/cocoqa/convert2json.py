import json, pickle



def convert(origin, save_path):
    ret = []
    with open(origin, "rb") as f:
        dic = pickle.load(f)
    for idx, inst in dic.items():
        tmp = {
            "question": inst["question"],
            "answer": inst["answer"],
            "image_id": inst["image_id"],
            "question_id": inst["question_id"]
        }
        ret.append(tmp)
    with open(save_path, "w") as f:
        json.dump(ret, f)



if __name__ == "__main__":
    train_split_dic_path = "/home/shiina/data/cocoqa/processed/train_split.pkl"
    val_split_dic_path = "/home/shiina/data/cocoqa/processed/val_split.pkl"
    test_split_dic_path = "/home/shiina/data/cocoqa/processed/test_split.pkl"
    train_save_path = "/home/shiina/shiina/question/iq/data/cocoqa/train_split.json"
    val_save_path = "/home/shiina/shiina/question/iq/data/cocoqa/val_split.json"
    test_save_path = "/home/shiina/shiina/question/iq/data/cocoqa/test_split.json"

    convert(train_split_dic_path, train_save_path)
    convert(val_split_dic_path, val_save_path)
    convert(test_split_dic_path, test_save_path)