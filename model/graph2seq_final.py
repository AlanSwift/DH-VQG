import torch
import torch.nn as nn
import torch.nn.functional as F
from module.encoder import str2encoder
from module.decoder import str2decoder
from module.submodule.loss import Graph2seqLoss, VisualHintLoss, VisualHintLossBalancedALL, KDLoss, VisualHintLossFocal, VisualHintLossHinge


class Graph2seqFinal(nn.Module):
    def __init__(self, cnn_encoder, cnn_backend, cnn_backend_path, fixed_block, dim_cnn, hidden_size,  # cnn encoder config
                 text_encoder, vocab, dim_word, encoder_style,
                 graph_encoder, dim_ppl, dim_loc_feats, dim_visual_hint, topk,
                 seq_decoder, text_max_length,
                 dropout, device, ppl_num, gnn, answer_amount,
                 epsilon=0.75
                 ):
        super(Graph2seqFinal, self).__init__()
        self._build(cnn_encoder=cnn_encoder, cnn_backend=cnn_backend, cnn_backend_path=cnn_backend_path,
                    fixed_block=fixed_block, dim_cnn=dim_cnn, hidden_size=hidden_size, dropout=dropout,
                    text_encoder=text_encoder, vocab=vocab, dim_word=dim_word, encoder_style=encoder_style,
                    graph_encoder=graph_encoder, dim_ppl=dim_ppl, dim_loc_feats=dim_loc_feats, dim_visual_hint=dim_visual_hint, topk=topk,
                    seq_decoder=seq_decoder, text_max_length=text_max_length, device=device, ppl_num=ppl_num, gnn=gnn,
                    answer_amount=answer_amount, epsilon=epsilon)
        self.loss_criteria = Graph2seqLoss(vocab=vocab)
        self.loss_visual_hint = VisualHintLossFocal(alpha=4, gamma=2)
        self.loss_vh_hinge = VisualHintLossHinge(alpha=4, gamma=2)
        self.loss_l2 = nn.MSELoss()
        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def _build(self, cnn_encoder, cnn_backend, cnn_backend_path, fixed_block, dim_cnn,
               text_encoder, vocab, dim_word, encoder_style,
               graph_encoder, dim_ppl, dim_loc_feats, dim_visual_hint, topk,
               seq_decoder, text_max_length,
               hidden_size, dropout, device, ppl_num, gnn, answer_amount, epsilon):
        self.cnn_encoder = str2encoder[cnn_encoder](cnn_backend, cnn_backend_path, fixed_block,
                                                    dim_cnn=dim_cnn, dim_out=hidden_size, dropout=dropout)
        self.text_encoder = str2encoder[text_encoder](vocab=vocab, input_size=dim_word, hidden_size=dim_word,
                                                      rnn_type="gru", rnn_dropout=dropout, device=device)
        self.graph_encoder = str2encoder[graph_encoder](dim_ppl=dim_ppl, dim_loc_feats=dim_loc_feats, topk=topk,
                dim_visual_hint=dim_visual_hint, dim_answer=dim_word, hidden_size=hidden_size, dropout=dropout,
                ppl_num=ppl_num, gnn=gnn, answer_amount=answer_amount, epsilon=epsilon)
        self.rnn_decoder = str2decoder[seq_decoder](max_step=text_max_length, vocab=vocab, dim_img=hidden_size,
                dim_word=dim_word, hidden_size=hidden_size, dropout=dropout, device=device, dim_ppl=hidden_size,
                                                    fuse_strategy="concat")

    @classmethod
    def from_opts(cls, args, vocab, device):
        return cls(cnn_encoder=args.cnn_encoder, cnn_backend=args.cnn_backend, cnn_backend_path=args.cnn_weight_path,

                   fixed_block=args.fixed_block, dim_cnn=args.cnn_out_dim, hidden_size=args.hidden_size,

                   text_encoder=args.text_encoder, vocab=vocab, dim_word=args.word_dim, encoder_style=args.encoder_style,

                   graph_encoder=args.graph_encoder, dim_ppl=args.proposal_dim, dim_loc_feats=args.loc_feats_dim,
                   dim_visual_hint=args.visual_hint_dim, topk=args.topk,
                   seq_decoder=args.seq_decoder, text_max_length=args.text_max_length,
                   dropout=args.dropout, device=device, ppl_num=args.ppl_num, gnn=args.gnn, answer_amount=args.answer_amount,
                   epsilon=args.epsilon)

    def forward(self, image: torch.Tensor, ppl_feats, ppl_info, visual_hint, question=None, answer=None,
                answer_idx=None, teacher_forcing=0):
        img_feats, img_feats_pool = self.cnn_encoder(image)
        answer_hint, answer_vec, answer_mask = self.text_encoder(answer, answer.sum(1))
        answer_hint = answer_hint.transpose(0, 1)
        answer_vec = answer_vec.transpose(0, 1).squeeze(1).contiguous()

        ppl_student, visual_hint_prob_student, pos_pred, ans_pred, adj = self.graph_encoder(ppl=ppl_feats, ppl_infos=ppl_info, answer_hint=answer_hint, answer_mask=answer_mask,
                                                                                       visual_hint=visual_hint, teacher_forcing=teacher_forcing,
                                                                                       img_feats=img_feats)

        pred = torch.argmax(visual_hint_prob_student, dim=-1)
        

        logits_student = self.rnn_decoder(img_feats=img_feats, ppl_feats=ppl_student, answer_hint=answer_vec, question=question, ppl_mask=pred) # , ppl_mask=pred

        step_cnt = logits_student.shape[1]
        if question is not None:
            loss_student, _ = self.loss_criteria(prob=logits_student, gt=question[:, 0:step_cnt].contiguous())

            loss_vh = self.loss_visual_hint(pred=visual_hint_prob_student, gt=visual_hint)
            loss_hinge = self.loss_vh_hinge(pred=visual_hint_prob_student, gt=visual_hint)

            pos_l2 = self.loss_l2(pos_pred, ppl_info[:, :, 0:4] * 10)
            ans_loss = self.loss_cross_entropy(ans_pred, answer_idx)

            return loss_student, loss_vh + loss_hinge, pos_l2, ans_loss
        else:
            return logits_student, visual_hint_prob_student, adj, ans_pred, pos_pred
