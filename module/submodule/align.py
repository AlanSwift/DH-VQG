import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionAlign(nn.Module):
    def __init__(self, query_size, memory_size, hidden_size, has_bias=False, attention_funtion="mlp", dropout=0.2):
        super(AttentionAlign, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.attn_type = attention_funtion
        assert self.attn_type in ["mlp", "general", "dot"]
        if self.attn_type == "mlp":
            self.query_in = nn.Linear(self.query_size, self.hidden_size, bias=True if has_bias else False)
            self.memory_in = nn.Linear(self.memory_size, self.hidden_size, bias=False)
        elif self.attn_type == "dot":
            if memory_size != query_size:
                raise ValueError("Parameter \"memory_size\" must be equal to \"query_size\"")
        elif self.attn_type == "general":
            self.query2memory = nn.Linear(query_size, memory_size, bias=False)
        self.out = nn.Linear(self.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.inf = 1e8

    def forward(self, query, memory, memory_mask=None, coverage=None):
        """
            Attention function
        Parameters
        ----------
        query: torch.Tensor, shape=[B, D]
        memory: torch.Tensor, shape=[B, N, D]
        memory_mask: torch.Tensor, shape=[B, N]
        coverage: torch.Tensor, shape=[B, N, D]

        Returns
        -------
        attn_results: torch.Tensor, shape=[B, D]
        attn_scores: torch.Tensor, shape=[B, N]
        """
        assert len(query.shape) == 2
        assert len(memory.shape) == 3
        assert query.shape[0] == memory.shape[0]
        if coverage is not None:
            assert len(coverage.shape) == 3
            assert coverage.shape[-1] == self.hidden_size
            assert coverage.shape == memory.shape

        aligns = self._calculate_aligns(query, memory, coverage=coverage)

        if memory_mask is not None:
            aligns = aligns.masked_fill(memory_mask == 0, -self.inf)
        return aligns

    def _calculate_aligns(self, src, tgt, coverage=None):
        """
            Attention score calculation.
        Parameters
        ----------
        src: torch.Tensor, shape=[B, D]
        tgt: torch.Tensor, shape=[B, N, D]
        coverage: torch.Tensor, shape=[B, N, D]
        Returns
        -------
        aligns: torch.Tensor, shape=[B, N]
        """
        if self.attn_type == "mlp":
            src = self.query_in(src)
            tgt = self.memory_in(tgt)
            aligns = src.unsqueeze(1) + tgt
            if coverage is not None:
                aligns += coverage
            aligns = torch.tanh(aligns)
            aligns = self.out(aligns)
            aligns = aligns.view(aligns.shape[0], aligns.shape[1])
        elif self.attn_type == "general" or self.attn_type == "dot":
            if coverage:
                assert tgt.shape == coverage.shape
                tgt = tgt + coverage
                tgt = torch.tanh(tgt)

            if self.attn_type == "general":
                src = self.query2memory(src)
            aligns = torch.bmm(src.unsqueeze(1), tgt.transpose(1, 2))
            aligns = aligns.squeeze(1).contiguous()
        else:
            raise NotImplementedError()
        return aligns




class AttentionPositionAlign(nn.Module):
    def __init__(self, query_size, memory_size, memory_step, hidden_size, has_bias=False, attention_funtion="mlp", dropout=0.2):
        super(AttentionPositionAlign, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.attn_type = attention_funtion
        self.memory_step = memory_step
        assert self.attn_type in ["mlp", "general"]
        if self.attn_type == "general":
            self.query2memory = nn.Linear(query_size, memory_size, bias=True)
            self.out = nn.Linear(memory_step, 4, bias=False)
        elif self.attn_type == "mlp":
            self.query_in = nn.Linear(self.query_size, self.hidden_size, bias=True if has_bias else False)
            self.memory_in = nn.Linear(self.memory_size, self.hidden_size, bias=False)
            self.out = nn.Linear(memory_step*self.hidden_size, 4, bias=False)
        else:
            raise NotImplementedError()


        self.dropout = nn.Dropout(dropout)
        self.inf = 1e8




    def forward(self, query, memory, memory_mask=None, coverage=None):
        """
            Attention function
        Parameters
        ----------
        query: torch.Tensor, shape=[B, M, D]
        memory: torch.Tensor, shape=[B, N, D]
        memory_mask: torch.Tensor, shape=[B, M, N]
        coverage: torch.Tensor, shape=[B, N, D]

        Returns
        -------
        attn_scores: torch.Tensor, shape=[B, M, N]
        """
        assert len(query.shape) == 3
        assert len(memory.shape) == 3
        assert query.shape[0] == memory.shape[0]
        if coverage is not None:
            assert len(coverage.shape) == 3
            assert coverage.shape[-1] == self.hidden_size
            assert coverage.shape == memory.shape

        aligns = self._calculate_aligns(query, memory, coverage=coverage)

        return aligns

    def _calculate_aligns(self, src, tgt, coverage=None):
        """
            Attention score calculation.
        Parameters
        ----------
        src: torch.Tensor, shape=[B, M, D]
        tgt: torch.Tensor, shape=[B, N, D]
        coverage: torch.Tensor, shape=[B, N, D]
        Returns
        -------
        aligns: torch.Tensor, shape=[B, M, N]
        """
        if self.attn_type == "general":
            tgt = torch.tanh(self.query2memory(tgt))
            aligns = torch.bmm(src, tgt.transpose(1, 2))
            aligns = F.relu(aligns)
            aligns = self.dropout(aligns)
            aligns = self.out(aligns)
        elif self.attn_type == "mlp":
            src = self.query_in(src)
            tgt = self.memory_in(tgt)
            aligns = src.unsqueeze(2) + tgt.unsqueeze(1)
            aligns = torch.relu(aligns)
            aligns = self.out(aligns.view(src.shape[0], src.shape[1], -1))
        else:
            raise NotImplementedError()
        return aligns