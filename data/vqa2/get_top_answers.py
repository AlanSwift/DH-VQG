import os, pickle

from collections import Counter

if __name__ == "__main__":

    train_filename = "/home/shiina/data/aaai/vqa2/train_split_dic_unique.pkl"
    answer_output = "/home/shiina/data/aaai/vqa2/top_answer_list.pkl"
    answer_list = []
    with open(train_filename, "rb") as f:
        content = pickle.load(f)
    for key_id, value in content.items():
        answer_list.append(value["answer"])

    counter = Counter(answer_list)

    most_common = counter.most_common(3062)
    most_common = (t for t, c in most_common)

    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)

    answers = tokens
    with open(answer_output, "wb") as f:
        pickle.dump(answers, f)

