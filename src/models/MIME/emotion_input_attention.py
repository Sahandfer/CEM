### MOST OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
## MINOR CHANGES
import torch
import torch.nn as nn
from src.models.MOEL.model import Encoder, Decoder


class EmotionInputEncoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        universal,
        emo_input,
    ):

        super(EmotionInputEncoder, self).__init__()
        self.emo_input = emo_input
        if self.emo_input == "self_att":
            self.enc = Encoder(
                2 * emb_dim,
                hidden_size,
                num_layers,
                num_heads,
                total_key_depth,
                total_value_depth,
                filter_size,
                universal=universal,
            )
        elif self.emo_input == "cross_att":
            self.enc = Decoder(
                emb_dim,
                hidden_size,
                num_layers,
                num_heads,
                total_key_depth,
                total_value_depth,
                filter_size,
                universal=universal,
            )
        else:
            raise ValueError("Invalid attention mode.")

    def forward(self, emotion, encoder_outputs, mask_src):
        if self.emo_input == "self_att":
            repeat_vals = [-1] + [encoder_outputs.shape[1] // emotion.shape[1]] + [-1]
            hidden_state_with_emo = torch.cat(
                [encoder_outputs, emotion.expand(repeat_vals)], dim=2
            )
            return self.enc(hidden_state_with_emo, mask_src)
        elif self.emo_input == "cross_att":
            return self.enc(encoder_outputs, emotion, (None, mask_src))[0]
