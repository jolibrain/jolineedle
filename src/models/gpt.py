"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import einops
from einops import rearrange, repeat
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from torch.nn import functional as F

from yolox.models import yolox_l, yolox_m, yolox_nano, yolox_s, yolox_tiny, yolox_x

from ..utils import CfgNode as CN
from .yolox import NeedleYOLOX
from .action_head import ActionHead


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )

        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward
        # self.mlpf = nn.Sequential(
        #     self.mlp["c_fc"],
        #     self.mlp["act"],
        #     self.mlp["c_proj"],
        #     self.mlp["dropout"],
        # )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = "gpt"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        self.block_size = config.block_size
        self.patch_size = config.patch_size
        self.image_processor = config.image_processor
        self.freeze_image_processor = config.freeze_image_processor
        self.image_cols = config.image_cols
        self.use_pos_emb = config.use_pos_emb
        self.no_patch_emb = config.no_patch_emb
        self.concat_emb = config.concat_emb
        self.decoder_pos_encoding = config.decoder_pos_encoding

        self.token_offset = 0

        # Set all dropout rates.
        config.embd_pdrop = config.resid_pdrop = config.attn_pdrop = config.dropout

        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict(
                {
                    # names follow the huggingface naming conventions
                    # GPT-1
                    "openai-gpt": dict(
                        n_layer=12, n_head=12, n_embd=768
                    ),  # 117M params
                    # GPT-2 configs
                    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                    "gpt2-medium": dict(
                        n_layer=24, n_head=16, n_embd=1024
                    ),  # 350M params
                    "gpt2-large": dict(
                        n_layer=36, n_head=20, n_embd=1280
                    ),  # 774M params
                    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                    # Gophers
                    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                    # (there are a number more...)
                    # I made these tiny models up
                    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
                    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
                    # beniz
                    "gpt-pico": dict(n_layer=2, n_head=2, n_embd=32),
                }[config.model_type]
            )

        # Heads.
        self.action_head = ActionHead(config.actions_info, config.n_embd)

        self.positional_encoding = PositionalEncoding2D(config.n_embd)
        if self.decoder_pos_encoding:
            self.decoder_token_pos_enc = PositionalEncoding1D(config.n_embd)

        self.embed_class = nn.Embedding(num_embeddings=100, embedding_dim=config.n_embd)
        config = deepcopy(config)
        self.token_offset += 1
        config.block_size += 1  # Conditional token.

        if self.concat_emb:
            n_embeddings = len(config.actions_info)  # token(s)
            n_embeddings += 1  # seq pos
            if not self.no_patch_emb:
                n_embeddings += 1  # patch
            if self.use_pos_emb:
                n_embeddings += 1  # patch position
            self.project_concat = nn.Linear(n_embeddings * config.n_embd, config.n_embd)

        # Load the detector.
        loader_yolox = {
            "yolox": yolox_nano,
            "yolox-nano": yolox_nano,
            "yolox-tiny": yolox_tiny,
            "yolox-s": yolox_s,
            "yolox-m": yolox_m,
            "yolox-l": yolox_l,
            "yolox-x": yolox_x,
        }
        base_model = loader_yolox[config.image_processor](
            pretrained=True, num_classes=80, device="cpu"
        )  # /!\ need to load the model with 80 classes in order to load the pretrained model!
        head = loader_yolox[config.image_processor](
            pretrained=False, num_classes=1, device="cpu"
        ).head
        self.yolox = NeedleYOLOX.from_base_yolox(
            base_model.backbone, head, config.detector_conf_threshold
        )

        if config.gpt_backbone:
            self.gpt_backbone = loader_yolox[config.gpt_backbone](
                pretrained=True, num_classes=80, device="cpu"
            ).backbone

        # Freeze the backbone.
        for param in self.yolox.backbone.parameters():
            if self.freeze_image_processor:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for param in head.parameters():
            param.requires_grad = True

        # Init lazy linear layer.
        if not self.no_patch_emb:
            if self.gpt_backbone:
                fpn_outs = self.gpt_backbone(
                    torch.randn(
                        1, config.n_channels, config.patch_size, config.patch_size
                    )
                )
            else:
                _, fpn_outs, _ = self.yolox(
                    torch.randn(
                        1, 1, config.n_channels, config.patch_size, config.patch_size
                    ),
                    torch.zeros(1, 1, 4, 6),  # fake targets
                )

            n_channels_fpn = fpn_outs[-1].shape[1]
            flatten_size = fpn_outs[-1].shape[2] * fpn_outs[-1].shape[3]
            self.embed_fpn = nn.Sequential(
                nn.Conv2d(
                    n_channels_fpn,
                    config.n_embd,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(flatten_size * config.n_embd, config.n_embd),
            )
            # sanity check
            self.embed_fpn(fpn_outs[-1])

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.actions_info[0].nclasses, config.n_embd),
                wpe=nn.Embedding(config.pos_emb_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        if self.decoder_pos_encoding:
            self.transformer["wpe"].requires_grad_(False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def embed_token_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens' positions so the decoder know the order of the tokens.

        Args:
            tokens: Batch of tokens.
                Shape of [batch_size, seq_len, hidden_size].

        Returns:
            Batch of tokens' positional embeddings.
                Shape of [batch_size, seq_len, hidden_size].
        """
        batch_size, seq_len, *_ = tokens.shape
        if self.decoder_pos_encoding:
            # Use 1D positional encoding.
            pos_emb = self.decoder_token_pos_enc(tokens)
        else:
            # Use learnable positional embeddings.
            pos_emb = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device)
            pos_emb = repeat(pos_emb, "s -> b s", b=batch_size)
            pos_emb = self.transformer.wpe(
                pos_emb
            )  # position embeddings of shape (b, t, n_embd)

        return pos_emb

    def embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed patches using the image model.

        Args:
            patches: Batch of patches.
                Shape of [batch_size, seq_len, n_channels, patch_size, patch_size].
            targets: Targets for the yolox model.
                See yolox for more details.

        Returns:
            Batch of patches' embeddings.
                Shape of [batch_size, seq_len, hidden_size].
        """
        seq_len = patches.shape[1]

        if hasattr(self, "gpt_backbone"):
            x = einops.rearrange(patches, "b t c w h -> (b t) c w h")

            # backpropagate throught standalone backbone
            fpn_outs = self.gpt_backbone(x)
        else:
            _, fpn_outs, _ = self.yolox(patches)

            # Do not backpropagate through yolox.
            fpn_outs = [fpn.detach() for fpn in fpn_outs]

        patch_emb = self.embed_fpn(fpn_outs[-1])
        patch_emb = rearrange(patch_emb, "(b s) h -> b s h", s=seq_len)
        return patch_emb

    def embed_patch_position(self, positions: torch.Tensor) -> torch.Tensor:
        """Use 2D positional encoding to embed patch positions.

        Args:
            positions: Batch of patches' positions.
                Shape of [batch_size, seq_len, 2].

        Returns:
            Batch of patches' positional embeddings.
                Shape of [batch_size, seq_len, hidden_size].
        """
        hidden_size = self.positional_encoding.org_channels
        pos_y, pos_x = positions[:, :, 0], positions[:, :, 1]
        map_pos_encodings = self.positional_encoding(
            torch.zeros(
                (
                    1,
                    pos_x.max() + 1,
                    pos_y.max() + 1,
                    hidden_size,
                ),
                device=positions.device,
            )
        )
        map_pos_encodings = map_pos_encodings.squeeze(0)  # [pos_x, pos_y, hiddens_size]
        positions = pos_y + pos_x * (pos_y.max() + 1)  # [batch_size, seq_len]
        map_pos_encodings = rearrange(map_pos_encodings, "x y h -> (x y) h")

        pos_encodings = map_pos_encodings[
            positions
        ]  # [batch_size, seq_len, hidden_size]
        return pos_encodings

    def forward(
        self,
        patches: torch.Tensor,
        actions: torch.LongTensor,
        classes: torch.Tensor,
        positions: Optional[torch.LongTensor] = None,
    ) -> tuple:
        """Do a forward pass of the model.
        Predicts the next actions and labels for each token.

        ---
        Args:
            patches: Consecutive patches of the image.
                Shape of [batch_size, n_tokens, n_channels, patch_size, patch_size].
            actions: Action taken before entering each patch.
                Shape of [batch_size, n_tokens].
                If multiple actions are taken, shape is [batch_size, n_tokens, n_action]
            classes: Label of the objects to look for in each batch.
                Shape of [batch_size,].
            positions: Tensor of coordinates (y, x) positions.
                Optional, used only if `self.use_pos_emb` is True.
                Shape of [batch_size, n_tokens, 2].

        ---
        Returns:
            The predicted action logits for each token.
                Shape of [batch_size, n_tokens, action_embed_size].
                If multiple actions are taken: [batch_size, n_tokens, n_actions, action_embed_size].
        """
        seq_len = actions.shape[1]
        assert (
            seq_len <= self.block_size
        ), f"Cannot forward sequence of length {seq_len}, block size is only {self.block_size}"
        assert (not self.use_pos_emb) or (positions is not None)

        # Embed all inputs. Each embedding is of shape [batch_size, seq_len, n_embd].
        embeddings = []
        if len(actions.shape) == 2:
            # single class action
            tok_emb = self.transformer.wte(actions)
            embeddings.append(tok_emb)
        elif len(actions.shape) == 3:
            # multiple actions class with same dimension
            for i in range(actions.shape[-1]):
                tok_emb = self.transformer.wte(actions[:, :, i])
                embeddings.append(tok_emb)

        pos_emb = self.embed_token_positions(tok_emb)
        embeddings.append(pos_emb)

        # This assumes that we use YOLOX as patch embeddings.
        if not self.no_patch_emb:
            patch_emb = self.embed_patches(patches)
            embeddings.append(patch_emb)

        if self.use_pos_emb:
            patch_pos_emb = self.embed_patch_position(positions)
            embeddings.append(patch_pos_emb)

        # Merge all embeddings into one token.
        if self.concat_emb:
            final_emb = torch.concat(
                embeddings, dim=2
            )  # [batch_size, seq_len, len(embeddings) * n_embd]
            final_emb = self.project_concat(final_emb)  # [batch_size, seq_len, n_embd]
        else:
            final_emb = torch.stack(
                embeddings, dim=2
            )  # [batch_size, seq_len, len(embeddings), n_embd]
            final_emb = torch.mean(final_emb, dim=2)  # [batch_size, seq_len, n_embd]

        # Add contitional token.
        class_emb = self.embed_class(classes)  # Shape of [batch_size, n_embd].
        class_emb = class_emb.unsqueeze(1)  # Shape of [batch_size, 1, n_embd].
        final_emb = torch.cat((class_emb, final_emb), dim=1)

        # forward the GPT model itself
        x = self.transformer.drop(final_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        action_logits = self.action_head(x)
        # Do not keep the first offset tokens.
        action_logits = action_logits[:, self.token_offset :].contiguous()

        return action_logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """Originally, this function separated the parameters into two groups,
        one with weight decay and one without.
        But now, we just separate the parameters that are not from YOLOX from the others.
        """
        optim_gpt = torch.optim.AdamW(
            params=[
                p for pn, p in self.named_parameters() if not pn.startswith("yolox")
            ],
            lr=train_config.learning_rate,
        )
        optim_yolox = torch.optim.AdamW(
            params=self.yolox.parameters(),
            lr=train_config.yolo_lr,
        )
        return optim_gpt, optim_yolox
