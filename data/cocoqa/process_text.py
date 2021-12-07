import argparse
import collections
import copy
import json
import os
import pickle
import random

from data.build_vocab import VocabBuilder
from data.vocab import Vocabulary
from utils.text import tokenize, stanfordcorenlp_pos_tag
from utils.calculate import cosine_similarity_2d, l2_similarity_2d
import nltk
from nltk.stem import WordNetLemmatizer
from stanfordcorenlp import StanfordCoreNLP
import tqdm

nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)
import torchtext.vocab as vocab
import torch, copy
glove = vocab.GloVe(name='6B', dim=300)
from utils.image import coco_class
id2cls = {idx: cls for idx, cls in enumerate(coco_class)}


def build_vocab(train_split_path, save_vocab_path):
    def parse_opt():
        parser = argparse.ArgumentParser()
        # # Data input settings
        parser.add_argument('--vocab-thresh', type=int,
                            default=3, help='')
        args = parser.parse_args()
        return args

    args = parse_opt()
    vocab_builder = VocabBuilder(vocab_thresh=args.vocab_thresh)

    def process_split(processor: VocabBuilder, split_path: str):

        with open(split_path, "rb") as f:
            split = pickle.load(f)

        for key, inst in split.items():
            question = inst["question"].lower()
            tokens = tokenize(question)
            processor.add2counters(tokens)
            answer = inst["answer"]
            answer = tokenize(answer)
            processor.add2vocabs(answer)

    process_split(vocab_builder, split_path=train_split_path)
    vocab_builder.finish()
    vocab_builder.process()
    vocab_builder.save(save_vocab_path)


def align_sentence_bbox(parsed_sentence, bbox_path, threshold=5.7):

    with open(bbox_path, "rb") as f:
        bbox = pickle.load(f)
    bbox_cls = bbox["box_cls"]

    # convert class to vectors
    class_vectors = []
    for cls_idx, cls in id2cls.items():
        cls_words_list = cls.split(" ")
        vectors = None
        for word in cls_words_list:
            vectors = glove.vectors[glove.stoi[word]] if vectors is None else vectors + glove.vectors[glove.stoi[word]]
        class_vectors.append(vectors / len(cls_words_list))

    bbox_cls_name = [coco_class[bbox_cls_id] for bbox_cls_id in bbox_cls]
    bbox_vectors = [class_vectors[bbox_cls_id].unsqueeze(0) for bbox_cls_id in bbox_cls]
    bbox_vectors = torch.cat(bbox_vectors, dim=0)

    parsed_sentence = copy.deepcopy(parsed_sentence)

    mask = torch.zeros(100)
    for word_inst in parsed_sentence:
        word = word_inst["word"]
        lemma = word_inst["lemma"]
        pos = word_inst["pos"]
        if pos not in ["PRP", "PRP$"] and pos[:1] != "N":
            continue

        if pos in ["PRP", "PRP$"]:
            if lemma in ["he", "she"]:
                word = "person"
                lemma = "person"
        if lemma in ["he", "she", "boy", "girl", "his", "her", "woman", "man", "people", "crowd", "lady", "gentleman"]:
            lemma = "person"

        if lemma in glove.stoi:
            vector = glove.vectors[glove.stoi[lemma]].unsqueeze(0)
            # sim = cosine_similarity_2d(vector, bbox_vectors)
            sim = l2_similarity_2d(vector, bbox_vectors)
            sim = sim.squeeze(0)
            valid_mask = sim <= threshold
            mask += valid_mask
    return mask > 0


def process_data():
    def parse_opt():
        parser = argparse.ArgumentParser()
        # # Data input settings
        parser.add_argument('--coco_cls', type=str, default='data/coco_class',
                            help='')


        parser.add_argument('--train-raw-json', type=str,
                            default="/home/shiina/data/cocoqa/raw/"
                                    'trainval_split.json',
                            help='')
        parser.add_argument('--train-image-dir', type=str, default='/home/shiina/data/coco2014/train2014/pic',
                            help='')
        parser.add_argument("--train-ppl-json-dir", type=str,
                            default="/home/shiina/data/aaai/coco_visual_features/train",
                            help="")

        parser.add_argument("--train-split-path", type=str,
                            default="/home/shiina/data/cocoqa/processed/"
                            'trainval_split.pkl',
                            help="")



        parser.add_argument('--test-raw-json', type=str,
                            default="/home/shiina/data/cocoqa/raw/"
                                    'test_split.json',
                            help='')
        parser.add_argument('--test-image-dir', type=str, default='/home/shiina/data/coco2014/val2014/pic',
                            help='')
        parser.add_argument("--test-ppl-json-dir", type=str,
                            default="/home/shiina/data/aaai/coco_visual_features/val",
                            help="")
        parser.add_argument("--test-split-path", type=str,
                            default="/home/shiina/data/cocoqa/processed/"
                                    'test_split.pkl',
                            help="")
        args = parser.parse_args()
        return args

    args = parse_opt()

    def make_split(raw_json_path, image_dir, ppl_dir, split):
        print("Making split: {}".format(split))
        ques_id2inst = collections.defaultdict()

        with open(raw_json_path, "r") as f:
            raw_insts = json.load(f)

        random.shuffle(raw_insts)

        if split == "train":
            image_path_format = "COCO_train2014_{}.jpg"
        elif split == "val":
            image_path_format = "COCO_val2014_{}.jpg"
        else:
            raise NotImplementedError()

        ret = {}
        cnt = 0

        unique_set = {}

        for inst in tqdm.tqdm(raw_insts):
            idx = inst["question_id"]
            image_id = inst["image_id"]
            image_id_pad = str(image_id).zfill(12)

            # filter dulplicate
            if inst["answer"].lower() == "":
                continue
            unique_key = (image_id_pad, inst["answer"].lower())
            if unique_set.get(unique_key) == 1:
                continue
            unique_set[unique_key] = 1

            image_filename = image_path_format.format(image_id_pad)
            image_path = os.path.join(image_dir, image_filename)
            ppl_json_path = os.path.join(ppl_dir, str(image_id_pad) + "_ppl.pkl")

            # check exist
            if not os.path.exists(image_path):
                raise RuntimeError("image not exist: {}".format(image_path))
            if not os.path.exists(ppl_json_path):
                raise RuntimeError("ppl not exist: {}".format(ppl_json_path))

            parsed_results = stanfordcorenlp_pos_tag(inst["question"].lower(), nlp_parser)
            if not parsed_results:
                continue

            question_valid_mask = align_sentence_bbox(parsed_results, ppl_json_path)
            question_tokens = [token_inst["word"] for token_inst in parsed_results]
            question_lemmas = [token_inst["lemma"] for token_inst in parsed_results]

            # answer_tokens = tokenize(inst["answer"].lower())
            parsed_results = stanfordcorenlp_pos_tag(inst["answer"].lower(), nlp_parser)
            if not parsed_results:
                continue
            answer_valid_mask = align_sentence_bbox(parsed_results, ppl_json_path)
            answer_tokens = [token_inst["word"] for token_inst in parsed_results]
            answer_lemmas = [token_inst["lemma"] for token_inst in parsed_results]

            valid_mask = (question_valid_mask.float() + answer_valid_mask.float()) > 0
            valid_mask = valid_mask.float()

            inst["image_path"] = image_path
            inst["ppl_json_path"] = ppl_json_path
            inst["question_tokens"] = question_tokens
            inst["question_lemmas"] = question_lemmas
            inst["answer_tokens"] = answer_tokens
            inst["answer_lemmas"] = answer_lemmas
            inst["visual_hint"] = valid_mask.numpy().tolist()
            if valid_mask.sum() == 0 and answer_tokens != ["no"]:
                # print(inst["question"], "-----", inst["answer"])
                # exit(0)
                continue
            else:
                ret[idx] = copy.deepcopy(inst)
        print("processing done. before: {}, after: {}".format(len(raw_insts), len(ret)))
        return ret


    train_dic = make_split(raw_json_path=args.train_raw_json,
                           image_dir=args.train_image_dir, ppl_dir=args.train_ppl_json_dir, split="train")
    with open(args.train_split_path, "wb") as f:
        pickle.dump(train_dic, f)
    val_dic = make_split(raw_json_path=args.test_raw_json,
                         image_dir=args.test_image_dir, ppl_dir=args.test_ppl_json_dir, split="val")
    with open(args.test_split_path, "wb") as f:
        pickle.dump(val_dic, f)


def combine_together():
    def parse_opt():
        parser = argparse.ArgumentParser()
        # # Data input settings
        parser.add_argument('--coco_cls', type=str, default='data/coco_class',
                            help='')

        parser.add_argument('--train-questions', type=str,
                            default='/home/shiina/data/cocoqa/train/'
                                    'questions.txt',
                            help='')
        parser.add_argument('--train-answer', type=str,
                            default='/home/shiina/data/cocoqa/train/'
                                    'answers.txt',
                            help='')
        parser.add_argument('--train-type', type=str,
                            default='/home/shiina/data/cocoqa/train/'
                                    'types.txt',
                            help='')
        parser.add_argument('--train-img-id', type=str,
                            default='/home/shiina/data/cocoqa/train/'
                                    'img_ids.txt',
                            help='')

        parser.add_argument('--train-image-dir', type=str, default='/home/shiina/data/coco2014/train2014/pic',
                            help='')
        parser.add_argument("--train-ppl-json-dir", type=str,
                            default="/home/shiina/data/aaai/coco_visual_features/train",
                            help="")

        parser.add_argument("--train-split-path", type=str,
                            default="/home/shiina/data/cocoqa/raw/"
                            'trainval_split.json',
                            help="")



        parser.add_argument('--test-questions', type=str,
                            default='/home/shiina/data/cocoqa/test/'
                                    'questions.txt',
                            help='')
        parser.add_argument('--test-answer', type=str,
                            default='/home/shiina/data/cocoqa/test/'
                                    'answers.txt',
                            help='')
        parser.add_argument('--test-type', type=str,
                            default='/home/shiina/data/cocoqa/test/'
                                    'types.txt',
                            help='')
        parser.add_argument('--test-img-id', type=str,
                            default='/home/shiina/data/cocoqa/test/'
                                    'img_ids.txt',
                            help='')

        parser.add_argument('--test-image-dir', type=str, default='/home/shiina/data/coco2014/val2014/pic',
                            help='')
        parser.add_argument("--test-ppl-json-dir", type=str,
                            default="/home/shiina/data/aaai/coco_visual_features/val",
                            help="")
        parser.add_argument("--test-split-path", type=str,
                            default="/home/shiina/data/cocoqa/raw/"
                                    'test_split.json',
                            help="")
        args = parser.parse_args()
        return args

    def read_file(file_path):
        with open(file_path, "r") as f:
            content = f.readlines()
            content = list(map(lambda x: x.strip().lower().replace("?", ""), content))
        return content

    def combine(question_path, answer_path, image_id_path, type_path, save_path, split):
        question = read_file(question_path)
        answer = read_file(answer_path)
        image_id = read_file(image_id_path)
        question_type = read_file(type_path)
        assert len(question) == len(answer)
        assert len(image_id) == len(answer)
        assert len(question_type) == len(answer)

        # assign a unique question id just like vqa2.0: img_id + %3d
        ques_cnt = {}
        item_collect = []
        inst_template = {"image_id": None, 'question': None, 'question_id': None, 'answer': None, "question_type": None}

        for ques, ans, img_id, ques_type in zip(question, answer, image_id, question_type):
            inst = copy.deepcopy(inst_template)
            cnt = ques_cnt.get(img_id, 0)
            ques_id = str(img_id) + str(cnt).zfill(3)
            inst["image_id"] = int(img_id)
            inst["question"] = ques
            inst["answer"] = ans
            inst["question_id"] = int(ques_id)
            inst["question_type"] = int(ques_type)
            item_collect.append(inst)
            ques_cnt[img_id] = cnt + 1
        with open(save_path, "w") as f:
            json.dump(item_collect, f)
        print("Done, example amount: {}".format(len(item_collect)))

    args = parse_opt()
    combine(question_path=args.train_questions, answer_path=args.train_answer,
            image_id_path=args.train_img_id, type_path=args.train_type,
            save_path=args.train_split_path, split="train")
    combine(question_path=args.test_questions, answer_path=args.test_answer,
            image_id_path=args.test_img_id, type_path=args.test_type,
            save_path=args.test_split_path, split="test")

def select(whole, keys):
    ret = {k: whole[k] for k in keys}
    return  ret

def split_trainval(trainval_path, ratio, save_train_path, save_val_path):
    with open(trainval_path, "rb") as f:
        trainval = pickle.load(f)
    keys = list(trainval.keys())
    print(keys[0], len(keys))
    random.shuffle(keys)

    train_num = int(len(keys) * (1 - ratio))
    train_keys = keys[:train_num]
    val_keys = keys[train_num:]

    train_split = select(trainval, train_keys)
    val_split = select(trainval, val_keys)

    with open(save_train_path, "wb") as f:
        pickle.dump(train_split, f)

    with open(save_val_path, "wb") as f:
        pickle.dump(val_split, f)
    pass

def get_answer_list(split_path, save_path):
    with open(split_path, "rb") as f:
        split = pickle.load(f)

    answer_list = []
    for key_id, value in split.items():
        answer_list.append(value["answer"])

    answers = list(set(answer_list))
    with open(save_path, "wb") as f:
        pickle.dump(answers, f)


if __name__ == "__main__":
    random.seed(1234)
    # combine_together()
    # process_data()
    # split_trainval(trainval_path="/home/shiina/data/cocoqa/processed/trainval_split.pkl",
    #                ratio=0.1,
    #                save_train_path="/home/shiina/data/cocoqa/processed/train_split.pkl",
    #                save_val_path="/home/shiina/data/cocoqa/processed/val_split.pkl")
    # get_answer_list(split_path="/home/shiina/data/cocoqa/processed/train_split.pkl",
    #                 save_path="/home/shiina/data/cocoqa/processed/answer_list.pkl")
    build_vocab(train_split_path="/home/shiina/data/cocoqa/processed/train_split.pkl",
                save_vocab_path="/home/shiina/data/cocoqa/processed/vqa2_vocab.json")
