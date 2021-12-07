import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from data.vocab import Vocabulary
from utils.image import RandomCrop
from transformers import BertTokenizer


class VQA2BertDataset(data.Dataset):
    def __init__(self, split_dic_path, image_size, image_crop_size, tokenizer,
                 prop_thresh=0, pad_length=20, verbase=1, ppl_num=100, split="train", ratio=1):
        super(VQA2BertDataset, self).__init__()
        with open(split_dic_path, "rb") as f:
            self.dic = pickle.load(f)
        self.split_idx2dic = {split_idx: idx for split_idx, idx in enumerate(self.dic.keys())}
        self.split = split
        self.ratio = ratio
        self.tokenizer = tokenizer

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
        question_idx = inst["question_id"]
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        img = self.Resize(img)
        img, cropped_shape = self.rand_crop.crop_image(img)
        img = self.img_process(img)

        # convert question to id
        question_str = inst["question"].lower().replace("?", "")

        answer = inst["answer_tokens"]
        answer_lemma = inst["answer_lemmas"]
        answer_ids = self.vocab.convert_tokens(answer, answer_lemma)
        answer_ids = np.array(answer_ids)
        ans_str = inst["answer"]
        if ans_str not in self.answer2id:
            # if self.split == "train":
            #     print("Warning: answer not found: ", ans_str)
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
        return img, box_feats, box_info, visual_hint, question, answer, answer_gt_idx, question_str, question_idx

    def __len__(self):
        return int(len(self.split_idx2dic)*self.ratio)

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
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = VQA2BertDataset("/home/shiina/data/aaai/vqa2/train_split_dic.pkl",
                          image_size=400,
                          image_crop_size=400,
                          tokenizer=tokenizer,
                          split="train", verbase=0)
    ds_loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=1)
    for data in ds_loader:
        print("------------------")
        img, box_feats, box_info, visual_hint, question, answer, answer_idx = data
        print(img.shape, "img1111")
        print(box_feats.shape, "box_feats1111")
        print(box_info.shape, "box_info1111")
        print(visual_hint.shape, "visual_hint_1111")
        print(question.shape, "question111")
        print(answer.shape, "answer11111")
        print(answer_idx.shape, "answeridx")
        print(answer_idx)
        exit(0)
