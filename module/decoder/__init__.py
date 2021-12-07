from module.decoder.seq_decoder import RNNDecoder


str2decoder = {"hie_rnn_decoder_base": RNNDecoder}

__all__ = ["str2decoder", "RNNDecoder"]
