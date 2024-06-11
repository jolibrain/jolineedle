import torch
from torch import nn
from dataclasses import dataclass

from ..env.common import ActionInfo

"""
Embed needle actions and then decode them
"""
# TODO Not needed for now
# class ActionEmbedder(nn.Module):


class ActionHead(nn.Module):
    def __init__(self, actions_info, n_embd):
        super().__init__()
        self.actions_info = actions_info
        # n_embd = config.n_embd
        # All actions are categorical for now
        self.lm_heads = nn.ModuleList(
            nn.Linear(n_embd, i.nclasses, bias=False) for i in actions_info
        )

    def forward(self, x):
        if len(self.lm_heads) == 1:
            action_logits = self.lm_heads[0](x)
        else:
            action_logits = []
            for lm_head in self.lm_heads:
                action_logits.append(lm_head(x))

            action_logits = torch.stack(action_logits, dim=2)
        return action_logits
