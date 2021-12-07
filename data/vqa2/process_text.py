import argparse
import collections
import copy
import json
import os
import pickle

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


def build_vocab():
    def parse_opt():
        parser = argparse.ArgumentParser()
        # # Data input settings
        parser.add_argument('--coco_cls', type=str, default='data/coco_class',
                            help='')
        parser.add_argument('--image-dir', type=str, default='/home/shiina/data/coco2014/val2014/pic',
                            help='')
        parser.add_argument('--output-json', type=str,
                            default='/home/shiina/shiina/question/aaai/data/vqa2/vqa2_vocab.json')
        parser.add_argument('--vocab-thresh', type=int,
                            default=3, help='')
        parser.add_argument('--train-questions', type=str,
                            default='/home/shiina/data/vqa2/v2_OpenEnded_mscoco_'
                                    'train2014_questions.json',
                            help='')
        parser.add_argument('--train-annotations', type=str,
                            default='/home/shiina/data/vqa2/v2_mscoco_'
                                    'train2014_annotations.json',
                            help='')
        parser.add_argument('--val-questions', type=str,
                            default='/home/shiina/data/vqa2/v2_OpenEnded_mscoco_'
                                    'val2014_questions.json',
                            help='')
        parser.add_argument('--val-annotations', type=str,
                            default='/home/shiina/data/vqa2/v2_mscoco_'
                                    'val2014_annotations.json',
                            help='')
        args = parser.parse_args()
        return args

    args = parse_opt()
    vocab_builder = VocabBuilder(vocab_thresh=args.vocab_thresh)

    def process_split(processor: VocabBuilder, question_json_path: str, annotations_json_path: str):

        with open(question_json_path, "r") as f:
            question_json = json.load(f)
        # print(question_json["questions"][0])
        for inst in question_json["questions"]:
            question = inst["question"].lower()
            tokens = tokenize(question)
            processor.add2counters(tokens)

        with open(annotations_json_path, "r") as f:
            anns_json = json.load(f)
        # print(anns_json['annotations'][0])
        for inst in anns_json['annotations']:
            answer = inst["multiple_choice_answer"]
            answer = tokenize(answer)
            processor.add2vocabs(answer)

    process_split(vocab_builder, args.train_questions, args.train_annotations)
    # process_split(vocab_builder, args.val_questions, args.val_annotations)
    vocab_builder.finish()
    vocab_builder.process()
    vocab_builder.save(args.output_json)


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

        parser.add_argument('--vocab-json-path', type=str,
                            default='/home/shiina/shiina/question/aaai/data/vqa2/vqa2_vocab.json')
        parser.add_argument('--train-questions', type=str,
                            default='/home/shiina/data/vqa2/v2_OpenEnded_mscoco_'
                                    'train2014_questions.json',
                            help='')
        parser.add_argument('--train-annotations', type=str,
                            default='/home/shiina/data/vqa2/v2_mscoco_'
                                    'train2014_annotations.json',
                            help='')
        parser.add_argument('--train-image-dir', type=str, default='/home/shiina/data/coco2014/train2014/pic',
                            help='')
        parser.add_argument("--train-ppl-json-dir", type=str,
                            default="/home/shiina/data/aaai/coco_visual_features/train",
                            help="")
        parser.add_argument("--train-split-path", type=str, default="/home/shiina/data/aaai/vqa2/train_split_dic_unique_full_7.pkl",
                            help="")
        parser.add_argument('--val-questions', type=str,
                            default='/home/shiina/data/vqa2/v2_OpenEnded_mscoco_'
                                    'val2014_questions.json',
                            help='')
        parser.add_argument('--val-annotations', type=str,
                            default='/home/shiina/data/vqa2/v2_mscoco_'
                                    'val2014_annotations.json',
                            help='')
        parser.add_argument('--val-image-dir', type=str, default='/home/shiina/data/coco2014/val2014/pic',
                            help='')
        parser.add_argument("--val-ppl-json-dir", type=str,
                            default="/home/shiina/data/aaai/coco_visual_features/val",
                            help="")
        parser.add_argument("--val-split-path", type=str, default="/home/shiina/data/aaai/vqa2/val_split_dic_unique_full_3.pkl",
                            help="")
        parser.add_argument("--threshold", type=float, default=7, help="")
        args = parser.parse_args()
        return args

    args = parse_opt()
    vocabulary = Vocabulary().load(args.vocab_json_path)

    def make_split(questions_path, annotations_path, vocab, image_dir, ppl_dir, split, threshold=5.7):
        print("Making split: {}".format(split))
        ques_id2inst = collections.defaultdict()
        with open(questions_path, "r") as f:
            question_json = json.load(f)
        for inst in question_json["questions"]:
            ques_id2inst[inst["question_id"]] = copy.deepcopy(inst)
        with open(annotations_path, "r") as f:
            annotation_json = json.load(f)
        for inst in annotation_json['annotations']:
            answer = inst["multiple_choice_answer"]
            ques_id = inst["question_id"]
            ques_id2inst[ques_id]["answer"] = answer

        if split == "train":
            image_path_format = "COCO_train2014_{}.jpg"
        elif split == "val":
            image_path_format = "COCO_val2014_{}.jpg"
        else:
            raise NotImplementedError()

        ret = {}
        cnt = 0

        unique_set = {}

        for idx, inst in tqdm.tqdm(ques_id2inst.items()):
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

            parsed_results = stanfordcorenlp_pos_tag(inst["question"].lower(), nlp_parser)
            if not parsed_results:
                continue

            question_valid_mask = align_sentence_bbox(parsed_results, ppl_json_path, threshold=threshold)
            question_tokens = [token_inst["word"] for token_inst in parsed_results]
            question_lemmas = [token_inst["lemma"] for token_inst in parsed_results]

            # answer_tokens = tokenize(inst["answer"].lower())
            parsed_results = stanfordcorenlp_pos_tag(inst["answer"].lower(), nlp_parser)
            if not parsed_results:
                continue
            answer_valid_mask = align_sentence_bbox(parsed_results, ppl_json_path, threshold=threshold)
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
                # print(inst["question"], inst["answer"])

                continue
            else:
                ret[idx] = copy.deepcopy(inst)
        print("processing done. before: {}, after: {}".format(len(ques_id2inst), len(ret)))
        return ret

    train_dic = make_split(args.train_questions, args.train_annotations, vocabulary,
                           args.train_image_dir, args.train_ppl_json_dir, split="train", threshold=args.threshold)
    with open(args.train_split_path, "wb") as f:
        pickle.dump(train_dic, f)
    # val_dic = make_split(args.val_questions, args.val_annotations, vocabulary,
    #                      args.val_image_dir, args.val_ppl_json_dir, split="val", threshold=args.threshold)
    # with open(args.val_split_path, "wb") as f:
    #     pickle.dump(val_dic, f)

def demo():
    raw_data = "James went to the corner-shop. He want to buy some (eggs), <milk> and bread for breakfast."
    props = {
        'annotators': 'ssplit,tokenize,pos',
        "tokenize.options":
            "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
        "tokenize.whitespace": False,
        'ssplit.isOneSentence': False,
        'outputFormat': 'json'
    }
    dep_json = nlp_parser.annotate(raw_data, props)
    print(dep_json)
    pass

if __name__ == "__main__":
    # demo()
    process_data()
    #build_vocab()
    pass
