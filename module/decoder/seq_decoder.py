import torch
import torch.nn as nn
import torch.nn.functional as F
from data.vocab import Vocabulary
from module.submodule.attention import Attention


class RNNDecoder(nn.Module):
    def __init__(self, max_step, vocab: Vocabulary, dim_img, dim_ppl, dim_word, hidden_size, dropout, device,
                 fuse_strategy="concat"):
        super(RNNDecoder, self).__init__()
        self.num_layers = 2
        self.hidden_size = hidden_size
        self.fuse_strategy = fuse_strategy
        self.vis_rnn = nn.LSTMCell(dim_img + dim_word, hidden_size)
        if self.fuse_strategy == "average" or self.fuse_strategy == "image":
            assert dim_img == dim_ppl
            self.lang_rnn = nn.LSTMCell(hidden_size + dim_img, hidden_size)
        elif self.fuse_strategy == "concat":
            self.lang_rnn = nn.LSTMCell(hidden_size + dim_img + dim_ppl, hidden_size)
        else:
            raise NotImplementedError()
        self.max_step = max_step
        self.vocab = vocab
        self.word_emb = nn.Sequential(nn.Embedding(len(vocab), dim_word),
                                      nn.ReLU(),
                                      nn.Dropout(dropout))
        self.device = device
        self.attn_image = Attention(query_size=hidden_size, memory_size=dim_img, hidden_size=hidden_size,
                                    has_bias=True, dropout=dropout)
        self.attn_ppl = Attention(query_size=hidden_size, memory_size=dim_ppl, hidden_size=hidden_size,
                                  has_bias=True, dropout=dropout)
        self.init_h = nn.Linear(dim_word, self.hidden_size)
        self.init_c = nn.Linear(dim_word, self.hidden_size)
        self.tanh = nn.Tanh()

        self.project = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, ppl_feats, answer_hint, ppl_mask=None, question=None, sampling=False):
        batch_size = img_feats.shape[0]
        rnn_state = self.init_hidden(batch_size, answer_hint)
        decoder_in = torch.zeros(batch_size).fill_(self.vocab.word2idx[self.vocab.SYM_SOS]).long().to(self.device)

        img_pool = img_feats.mean(dim=1)
        # img_pool, _ = self.attn_pool(query=answer_hint, memory=img_feats)
        decoder_out_collect = []
        use_teacher_forcing = question is not None

        for idx in range(self.max_step):
            if all(decoder_in == self.vocab.word2idx[self.vocab.SYM_PAD]):
                break
            dec_emb = self.word_emb(decoder_in)
            decoder_out, rnn_state = self._decode_step(dec_input_emb=dec_emb, rnn_state=rnn_state, image_feats=img_feats, image_pool=img_pool,
                                                       ppl_feats=ppl_feats, answer_hint=answer_hint, ppl_mask=ppl_mask)
            decoder_out_logits = self.project(decoder_out)
            logprob = torch.log_softmax(decoder_out_logits, dim=-1)
            decoder_out_collect.append(logprob.unsqueeze(1))

            if use_teacher_forcing:
                decoder_in = question[:, idx]
            else:
                decoder_in = logprob.argmax(dim=-1)
        decoder_ret = torch.cat(decoder_out_collect, dim=1)
        return decoder_ret


    def _decode_step(self, dec_input_emb, rnn_state, image_feats, image_pool, ppl_feats, answer_hint, ppl_mask=None):
        visual_lstm_input = torch.cat((image_pool, dec_input_emb), dim=-1)
        h_vis, c_vis = self.vis_rnn(visual_lstm_input, (rnn_state[0][0], rnn_state[1][0]))
        image_attn, _ = self.attn_image(query=h_vis, memory=image_feats)
        ppl_attn, _ = self.attn_ppl(query=h_vis, memory=ppl_feats, memory_mask=ppl_mask)
        if self.fuse_strategy == "average":
            language_lstm_input = torch.cat((image_attn + ppl_attn, h_vis), dim=-1)
        elif self.fuse_strategy == "concat":
            language_lstm_input = torch.cat((image_attn, ppl_attn, h_vis), dim=-1)
        elif self.fuse_strategy == "image":
            language_lstm_input = torch.cat((image_attn, h_vis), dim=-1)
        else:
            raise NotImplementedError()
        h_lang, c_lang = self.lang_rnn(language_lstm_input, (rnn_state[0][1], rnn_state[1][1]))
        rnn_results = self.dropout(h_lang)
        rnn_state = (torch.stack([h_vis, h_lang]), torch.stack([c_vis, c_lang]))
        return rnn_results, rnn_state

    def init_hidden(self, bsz, answer_vector):
        c = self.init_c(answer_vector)
        c = self.tanh(c)

        h = self.init_h(answer_vector)
        h = self.tanh(h)
        return h.unsqueeze(0).expand(2, -1, -1), c.unsqueeze(0).expand(2, -1, -1)
