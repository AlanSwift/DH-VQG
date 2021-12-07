import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from data.vocab import Vocabulary
from utils.image import RandomCrop


class COCOQADataset(data.Dataset):
    def __init__(self, split_dic_path, vocab_path, answer_path, image_size, image_crop_size,
                 prop_thresh=0, pad_length=20, verbase=1, ppl_num=100, split="train"):
        super(COCOQADataset, self).__init__()
        with open(split_dic_path, "rb") as f:
            self.dic = pickle.load(f)
        self.split_idx2dic = {split_idx: idx for split_idx, idx in enumerate(self.dic.keys())}
        self.split = split
        with open(answer_path , "rb") as f:
            answer_dict_list = pickle.load(f)
            self.answer2id = {ans: ans_id for ans_id, ans in enumerate(answer_dict_list)}
            self.answer2id["UNK"] = len(answer_dict_list)
            print("Answer number", len(answer_dict_list) + 1) # 16367
        self.vocab = Vocabulary.load(vocab_path)
        self.Resize = transforms.Resize((image_size, image_size))
        self.rand_crop = RandomCrop(image_crop_size)

        self.img_process = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.prop_thresh = prop_thresh
        self.pad_length = pad_length
        self.verbase = verbase
        self.ppl_num = ppl_num

    def __getitem__(self, item):
        idx = self.split_idx2dic[item]
        inst = self.dic[idx]
        if self.verbase > 0:
            print(inst)
        image_path = inst["image_path"]
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        img = self.Resize(img)
        img, cropped_shape = self.rand_crop.crop_image(img)
        img = self.img_process(img)

        # load proposals
        ppl_json = pickle.load(open(inst['ppl_json_path'], "rb"))
        # print(ppl_json.keys())
        ori_height = ppl_json["height"]
        ori_width = ppl_json["width"]
        num_boxes = ppl_json["num_boxes"]
        box_scores = ppl_json["box_scores"]
        box_cls = ppl_json["box_cls"]
        box_feats = ppl_json["box_feats"]
        box_coordinate = ppl_json["boxes_coordinate"]
        box_cls = box_cls.reshape(num_boxes, 1)

        box_infos = np.concatenate((box_coordinate, box_cls.reshape(num_boxes, 1), box_scores.reshape(num_boxes, 1)),
                                   axis=1)
        assert ori_height == height
        assert ori_width == width
        # print(ori_height, ori_width, num_boxes, box_scores.shape, box_cls.shape, box_feats.shape)

        box_coordinate = self.rand_crop.crop_bbox(box_coordinate, cropped_shape)
        box_infos[:, :4] = box_coordinate
        box_infos[:, 0] /= width
        box_infos[:, 2] /= width
        box_infos[:, 1] /= height
        box_infos[:, 3] /= height

        fg_ppl = box_infos[:, 5] >= self.prop_thresh
        fg_idx = np.nonzero(fg_ppl)[0]
        box_feats = box_feats[fg_idx, :]
        box_infos = box_infos[fg_idx, :]
        box_cls = box_cls[fg_idx]

        # visual hint
        visual_hint = np.array(inst["visual_hint"])
        visual_hint = visual_hint[fg_idx]

        # convert question to id
        question = inst["question_tokens"]
        question_lemma = inst["question_lemmas"]
        question_ids = self.vocab.convert_tokens(question, question_lemma)
        question_ids = np.array(question_ids)
        question_str = inst["question"].lower().replace("?", "")

        answer = inst["answer_tokens"]
        answer_lemma = inst["answer_lemmas"]
        answer_ids = self.vocab.convert_tokens(answer, answer_lemma)
        answer_ids = np.array(answer_ids)
        ans_str = inst["answer"]
        if ans_str not in self.answer2id:
            if self.split == "train":
                print("Warning: answer not found: ", ans_str)
            answer_gt_idx = self.answer2id["UNK"]
        else:
            answer_gt_idx = self.answer2id[ans_str]


        box_feats = box_feats[:self.ppl_num, :]
        box_infos = box_infos[:self.ppl_num, :]
        box_cls = box_cls[:self.ppl_num]
        visual_hint = visual_hint[:self.ppl_num]


        if self.verbase > 0:
            print("image", img.shape, type(img))
            print("box_feats", box_feats.shape, type(box_feats))
            print("box_infos", box_infos.shape, type(box_infos))
            print("box_cls", box_cls.shape, type(box_cls))
            print("question ids", question_ids.shape, type(question_ids))
            print("answer_ids", answer_ids.shape, type(answer_ids))
        img, box_feats, box_info, question, answer, visual_hint = self.vectorize(img=img, box_feats=box_feats,
                                                                                 box_infos=box_infos,
                                                                                 box_cls=box_cls, question=question_ids,
                                                                                 answer=answer_ids,
                                                                                 visual_hint=visual_hint)
        return img, box_feats, box_info, visual_hint, question, answer, answer_gt_idx, question_str

    def __len__(self):
        return int(len(self.split_idx2dic))

    def vectorize(self, img, box_feats, box_infos, box_cls, question, answer, visual_hint):
        box_feats = torch.from_numpy(box_feats)
        box_infos = torch.from_numpy(box_infos)
        box_cls = torch.from_numpy(box_cls)
        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
        visual_hint = torch.from_numpy(visual_hint)
        # if (visual_hint == 0).all():
        #     visual_hint = torch.ones_like(visual_hint)

        box_info = torch.cat((box_infos.float(), box_cls.float()), dim=-1)

        def pad(x, length=self.pad_length):
            assert len(x.shape) == 1
            assert isinstance(x, torch.Tensor)
            pad_len = length - x.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len).fill_(self.vocab(self.vocab.SYM_PAD))
                x = torch.cat((x, pad.long()), dim=0)
            elif pad_len <= 0:
                x = x[:length]
            return x

        question = pad(question)
        answer = pad(answer)

        return img, box_feats, box_info, question, answer, visual_hint


if __name__ == "__main__":
    dataset = COCOQADataset("/home/shiina/data/cocoqa/processed/train_split.pkl",
                          answer_path="/home/shiina/data/cocoqa/processed/answer_list.pkl",
                          vocab_path="/home/shiina/data/cocoqa/processed/vqa2_vocab.json",
                          image_size=400,
                          image_crop_size=400,
                          split="train", verbase=0)

    ds_loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=1)
    for data in ds_loader:
        print("------------------")
        img, box_feats, box_info, visual_hint, question, answer, answer_idx, question_str = data
        print(img.shape, "img1111")
        print(box_feats.shape, "box_feats1111")
        print(box_info.shape, "box_info1111")
        print(visual_hint.shape, "visual_hint_1111")
        print(question.shape, "question111")
        print(answer.shape, "answer11111")
        print(answer_idx.shape, "answeridx")
        print(answer_idx)
        print(question_str)
        exit(0)
