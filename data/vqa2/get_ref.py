import pickle, json


def process(val_split_path, save_path):
    with open(val_split_path, "rb") as f:
        content = pickle.load(f)
    id2ref = {}
    for idx, item in content.items():
        inst = {"image_name": item["image_id"], "answer": item["answer"], "question": item["question"]}
        id2ref[idx] = inst
    with open(save_path, "w") as f:
        json.dump(id2ref, f)


if __name__ == "__main__":
    process(val_split_path="/home/shiina/data/aaai/vqa2/val_split_dic_unique.pkl",
            save_path="/home/shiina/data/aaai/vqa2/human_ref.json")
