import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from utils.tools import get_log
from data.vqa2.data_loader_answer_final import VQA2Dataset
from model import str2model
import time
import numpy as np
from module.evaluate import str2metric
import torch.distributed as dist
import torch.utils.data.distributed
import pickle, json
from utils.calculate import bbox_overlaps_batch
import math


class TrainerFinal:
    def __init__(self, args, inference=False):
        super(TrainerFinal, self).__init__()
        self.verbase = 1
        self.device = torch.device("cuda:0")
        self.opt = args
        print("*********[Trainer configure]***********")
        print(self.opt)
        self.distributed = self.opt.world_size > 1 and not inference

        if self.distributed:
            dist.init_process_group(backend=self.opt.dist_backend, init_method=self.opt.dist_url,
                                    world_size=self.opt.world_size, rank=self.opt.rank)
        self.inference = inference

        self.__build_logger(args.log_path)
        self.__build_dataloader(args)
        self.__build_model(args)
        self.__build_optimizer(args)
        self.__build_evaluator(args)

    def __build_logger(self, log_path):
        if not os.path.exists("log"):
            os.mkdir("log")
        self.logger = get_log(log_path)

    def __build_dataloader(self, args):
        # train dataloader
        train_dataset = VQA2Dataset(split_dic_path=args.train_split_dic_path, vocab_path=args.vocab_path,
                                    image_size=args.image_size, image_crop_size=args.image_crop_size, split="train",
                                    prop_thresh=0, pad_length=args.text_max_length, verbase=0, ppl_num=args.ppl_num,
                                    answer_path=args.answer_path)
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=self.train_sampler,
                                           shuffle=shuffle, num_workers=args.num_workers)
        self.vocab = train_dataset.vocab

        # val dataloader
        val_dataset = VQA2Dataset(split_dic_path=args.val_split_dic_path, vocab_path=args.vocab_path,
                                  image_size=args.image_crop_size, image_crop_size=args.image_crop_size, split="val",
                                  prop_thresh=0, pad_length=args.text_max_length, verbase=0, ppl_num=args.ppl_num,
                                  answer_path=args.answer_path)

        self.val_sampler = None

        self.val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=self.val_sampler,
                                         shuffle=False, num_workers=args.num_workers)

        # test dataloader
        test_dataset = VQA2Dataset(split_dic_path=args.test_split_dic_path, vocab_path=args.vocab_path,
                                  image_size=args.image_crop_size, image_crop_size=args.image_crop_size, split="test",
                                  prop_thresh=0, pad_length=args.text_max_length, verbase=0, ppl_num=args.ppl_num,
                                  answer_path=args.answer_path)

        self.test_sampler = None

        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=self.test_sampler,
                                          shuffle=False, num_workers=args.num_workers)

    def __build_model(self, args):
        model = str2model[args.model].from_opts(args, vocab=self.vocab, device=self.device)
        model = model.to(self.device)
        if self.distributed:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        self.model = model

    def __build_optimizer(self, args):
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if 'cnn' in key:
                    params += [{'params': [value], 'lr': float(args.cnn_learning_rate),
                                'weight_decay': float(args.cnn_weight_decay),
                                'betas': (float(args.cnn_optim_alpha), float(args.cnn_optim_beta))}]
                else:
                    params += [{'params': [value], 'lr': float(args.learning_rate),
                                'weight_decay': float(args.weight_decay),
                                'betas': (float(args.optim_alpha), float(args.optim_beta))}]
        self.optimizer = optim.Adam(params)
        assert args.lr_scheduler == "ExponentialLR"
        self.lr_scheduler: optim.lr_scheduler.ExponentialLR = getattr(optim.lr_scheduler, args.lr_scheduler)(self.optimizer, gamma=args.gamma)

    def __build_evaluator(self, args):
        self.metric = [str2metric["cider"](df="corpus"),
                       str2metric["bleu"](n_grams=[1, 2, 3, 4]),
                       str2metric["meteor"](),
                       str2metric["rouge"](),
                       str2metric["spice"](),
                       str2metric["accuracy"](metrics=["precision", "recall", "F1", "accuracy"])]

    def save(self, epoch, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        args_filename = "args-{}.pkl".format(epoch)
        model_name = "model-{}.pth".format(epoch)
        with open(os.path.join(save_dir, args_filename), "wb") as f:
            pickle.dump(self.opt, f)

        torch.save(self.model.module.state_dict(), os.path.join(save_dir, model_name))

    def load(self, epoch, save_dir):
        model_name = "model-{}.pth".format(epoch)
        checkpoint_path = os.path.join(save_dir, model_name)
        self.model.load_state_dict(torch.load(checkpoint_path))

    def average_gradients(self, model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size
    
    def weight_decay(self, epoch):
        if epoch < 6:
            return 0
        epoch = epoch - 6
        return (1/(1 + math.exp(-(epoch/2))) - 0.5) * 2

    def train_epoch(self, epoch):
        if self.distributed:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        start = time.time()
        loss_collect_student = []
        vh_student_loss_collect = []
        vh_teacher_loss_collect = []
        recover_loss_collect = []
        pos_loss_collect = []
        ans_loss_collect = []

        for step, data in enumerate(self.train_dataloader):
            img, box_feats, box_info, visual_hint, question, answer, answer_idx, question_str, _ = self.to_cuda(data)

            model_in = {"image": img, "ppl_feats": box_feats, "ppl_info": box_info, "visual_hint": visual_hint,
                        "question": question, "answer": answer, "answer_idx": answer_idx, "teacher_forcing": self.opt.graph_enc_teacher_forcing}

            loss_student, loss_vh_student, pos_l2, ans_loss = self.model(**model_in)

            pernal = 1.0

            loss_all = loss_student + loss_vh_student*self.opt.vh_weight*pernal + pos_l2*self.opt.pos_weight*pernal + ans_loss*self.opt.ans_weight*pernal

            self.optimizer.zero_grad()
            loss_all.backward()
            if self.opt.world_size > 1:
                self.average_gradients(self.model)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

            loss_collect_student.append(loss_student.item())

            vh_student_loss_collect.append(loss_vh_student.item())
            pos_loss_collect.append(pos_l2.item())
            ans_loss_collect.append(ans_loss.item())

            if step % 100 == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "step {}/{} (epoch {}), lm_loss_student = {:.4f}, vh_student_loss = {:.4f}, "
                    "pos_loss = {:.4f}, ans_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
                    .format(step, len(self.train_dataloader), epoch, np.mean(loss_collect_student),
                    np.mean(vh_student_loss_collect), np.mean(pos_loss_collect),
                    np.mean(ans_loss_collect), float(self.opt.learning_rate), end - start))
                loss_collect_student = []
                vh_student_loss_collect = []
                pos_loss_collect = []
                ans_loss_collect = []
                start = time.time()

    @torch.no_grad()
    def evaluate(self, epoch=None, split="val", save_dir="results"):
        self.model.eval()
        start = time.time()

        assert split in ["val", "test"]

        dataloader = self.val_dataloader if split == "val" else self.test_dataloader

        pred_collect = []
        gt_collect = []

        vh_pred_teacher_collect = []
        vh_pred_student_collect = []
        l2_loss_collect = []
        vh_gt_collect = []
        pred_vh = []
        pred_adj = []
        vis_dump = []

        cnt = 0
        dump_collect = []
        ans_correct_collect = []
        ans_all_collect = []

        pos_iou_collect = []

        with torch.no_grad():
            for step, data in enumerate(dataloader):
                if step % 100 == 0 and step != 0:
                    print("Step: {}/{}".format(step, len(dataloader)))
                cnt += 1
                img, box_feats, box_info, visual_hint, question, answer, answer_idx, question_gt, question_idx = self.to_cuda(data)
                model_in = {"image": img, "ppl_feats": box_feats, "ppl_info": box_info, "visual_hint": visual_hint,
                            "question": None, "answer_idx":None, "answer": answer}
                prob, vh_student_prob, adj, ans_pred, pos_pred = self.model(**model_in)
                ans_pred_idx = ans_pred.argmax(-1)
                correct = ans_pred_idx == answer_idx
                ans_correct_collect.append(correct.float().sum())
                ans_all_collect.append(float(answer_idx.shape[0]))


                pos_pred = pos_pred.abs()
                overlaps = bbox_overlaps_batch(pos_pred, box_info[:, :, :4] * 10)
                pos_iou_collect.extend(overlaps.view(-1).detach().cpu().numpy().tolist())

                pred_adj = adj.detach().cpu().numpy()

                prob_ids = prob.argmax(dim=-1)
                sentence_pred = self.vocab.convert_ids(prob_ids.detach().cpu().data)
                sentence_gt = question_gt
                pred_collect.extend(sentence_pred)
                gt_collect.extend(sentence_gt)

                vh_prob_student = vh_student_prob.argmax(dim=-1)

                # prob = torch.softmax(vh_student_prob, dim=-1)
                # pos = prob[:, :, 1]
                # vh_prob_student = (pos > 0.55).float()

                vh_pred_student_collect.append(vh_prob_student.detach().cpu())
                vh_gt_collect.append(visual_hint.detach().cpu())

                for i in range(prob.shape[0]):
                    item = {"prediction": sentence_pred[i], "gt": sentence_gt[i], "question_id": question_idx[i].item()}
                    dump_collect.append(item)
                vh_pred = vh_prob_student.detach().cpu().numpy()
                vh_gt = visual_hint.detach().cpu().numpy()
                for i in range(prob.shape[0]):
                    item = {"question_id": question_idx[i].item(), "vh_pred": vh_pred[i, :], "vh_gt": vh_gt[i, :],
                            "adj": pred_adj[i, :, :]}
                    vis_dump.append(item)
            end = time.time()

            if epoch is not None:
                os.makedirs(exist_ok=True, name=save_dir)
                with open(os.path.join(save_dir, "vis-{}.json".format(epoch)), "w") as f:
                    json.dump(dump_collect, f)
                    print("Prediction saved in {}".format(os.path.join(save_dir, "vis-{}.json".format(epoch))))
                with open(os.path.join(save_dir, "vis-{}.pkl".format(epoch)), "wb") as f:
                    pickle.dump(vis_dump, f)
                    print("Visualization obj saved in {}".format(os.path.join(save_dir, "vis-{}.pkl".format(epoch))))

            self.logger.info("*********** Evaluation, split: {} ***********".format(split))
            self.logger.info("Time cost: {:.4f}".format(end - start))

            score, scores = self.metric[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
            print(scores)
            self.logger.info("Metric {}: {:.4f}".format(str(self.metric[0]), score))

            score, _ = self.metric[1].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
            self.logger.info("Metric {}: @1 - {:.4f}, @2 - {:.4f}, @3 - {:.4f}, @4 - {:.4f}".format(
                str(self.metric[1]), score[0], score[1], score[2], score[3]))
            bleu4 = score[3]

            score, _ = self.metric[2].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
            self.logger.info("Metric {}: {:.4f}".format(str(self.metric[2]), score))

            score, _ = self.metric[3].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
            self.logger.info("Metric {}: {:.4f}".format(str(self.metric[3]), score))

            score, _ = self.metric[4].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
            self.logger.info("Metric {}: {:.4f}".format(str(self.metric[4]), score))


            # visual hint evaluation

            vh_pred_student_collect_concat = torch.cat(vh_pred_student_collect, dim=0).view(-1).int()
            vh_gt_collect_concat = torch.cat(vh_gt_collect, dim=0).view(-1).int()
            scores = self.metric[5].calculate_scores(ground_truth=vh_gt_collect_concat,
                                                     predict=vh_pred_student_collect_concat)
            self.logger.info("Visual Hint Prediction performance")
            self.logger.info("Metric Precision: 0 - {}, 1 - {}".format(scores[0][0], scores[0][1]))
            self.logger.info("Metric Recall: 0 - {}, 1 - {}".format(scores[1][0], scores[1][1]))
            self.logger.info("Metric F1: 0 - {}, 1 - {}".format(scores[2][0], scores[2][1]))
            self.logger.info("Metric Accuracy: {}".format(scores[3]))

            self.logger.info("------------------------------------")
            self.logger.info("Mean Accuracy: {}".format(np.sum(ans_correct_collect) / np.sum(ans_all_collect)))
            self.logger.info("Mean IOU: {}".format(np.mean(pos_iou_collect)))
        return bleu4

    def to_cuda(self, data):
        ret = []
        for x in data:
            if isinstance(x, torch.Tensor):
                ret.append(x.to(self.device))
            else:
                ret.append(x)
        return ret

    def train(self):
        self.logger.info("Start training: vh_weight: {:.4f}, pos_weight: {:.4f}, ans_weight: {:.4f}".
                         format(self.opt.vh_weight, self.opt.pos_weight, self.opt.ans_weight))

        for epoch in range(self.opt.epoch_all):
            if epoch > self.opt.lr_decay_epoch_start and epoch % self.opt.lr_decay_epoch_num == 0:
                if float(self.opt.learning_rate) < 5e-5:
                    continue
                self.lr_scheduler.step()
                self.opt.learning_rate = self.optimizer.param_groups[-1]['lr']
            self.train_epoch(epoch)
            if self.distributed and dist.get_rank() == 0:
                self.save(save_dir=self.opt.checkpoint_path, epoch=epoch)
            elif not self.distributed:
                self.save(save_dir=self.opt.checkpoint_path, epoch=epoch)
            # dist.barrier()
            if self.distributed and dist.get_rank() == 0:
                self.evaluate(split="val")
            if self.distributed and dist.get_rank() == 1:
                self.evaluate(epoch=epoch, split="test", save_dir=self.opt.save_dir)
            dist.barrier()


