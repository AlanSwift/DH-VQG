import os, pickle

if __name__ == "__main__":
    train_filename = "/home/shiina/data/aaai/vqa2/train_split_dic_unique_full_7.pkl"
    answer_output = "/home/shiina/data/aaai/vqa2/answer_list_7.pkl"

    # train_filename = "/home/shiina/data/aaai/vqa2/train_split_dic_unique.pkl"
    # answer_output = "/home/shiina/data/aaai/vqa2/answer_list.pkl"
    answer_list = []
    with open(train_filename, "rb") as f:
        content = pickle.load(f)
    for key_id, value in content.items():
        answer_list.append(value["answer"])
    print(len(answer_list))

    answers = list(set(answer_list))
    with open(answer_output, "wb") as f:
        pickle.dump(answers, f)

