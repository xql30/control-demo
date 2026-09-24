from dataclasses import dataclass
import copy
from logging import getLogger
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.category_router import HierarchicalCategoryRouter
from genrec.model import AbstractModel
from genrec.sid_anchor_router import SIDAnchorRouter
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import log
from LLaDA import LLaDAConfig, LLaDAModelLM
from LLaDA.modeling_llada import (CausalLMOutputWithPast, ModuleType,
                                  init_weights)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


@dataclass
class LLaDARecOutPut:
    loss: torch.Tensor
    his_mask_loss: torch.Tensor
    target_mask_loss: torch.Tensor
    qwen_distill_loss: Optional[torch.Tensor] = None
    user_code_aux_loss: Optional[torch.Tensor] = None
    user_code_mask_loss: Optional[torch.Tensor] = None
    rec_control_pretrain_loss: Optional[torch.Tensor] = None
    rec_control_align_loss: Optional[torch.Tensor] = None
    rec_control_item_rank_loss: Optional[torch.Tensor] = None
    rec_control_sid_token_loss: Optional[torch.Tensor] = None
    semantic_bridge_loss: Optional[torch.Tensor] = None
    multi_intent_memory_loss: Optional[torch.Tensor] = None
    probabilistic_intent_counterfactual_loss: Optional[torch.Tensor] = None
    probabilistic_intent_router_loss: Optional[torch.Tensor] = None
    collaborative_intent_counterfactual_loss: Optional[torch.Tensor] = None
    sid_anchor_router_loss: Optional[torch.Tensor] = None
    sid_anchor_counterfactual_loss: Optional[torch.Tensor] = None
    sid_anchor_consistency_loss: Optional[torch.Tensor] = None
    sid_anchor_item_contrastive_loss: Optional[torch.Tensor] = None


class LLaDARec(AbstractModel):

    def __init__(self, config: dict, dataset: AbstractDataset,
                 tokenizer: AbstractTokenizer):
        super(LLaDARec, self).__init__(config, dataset, tokenizer)

        self.logger = getLogger()

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        self.posValidTokens = self.get_posValidToken().to(
            self.config['device'])
        self.sid_trie_children = self._build_sid_trie_children()

        self.mask_token_id = tokenizer.mask_token
        lladaconfig = LLaDAConfig(
            activation_type=config['activation_type'],
            attention_dropout=config['dropout_rate'],
            bias_for_layer_norm=config['bias_for_layer_norm'],
            block_type=config['block_type'],
            d_model=config['d_model'],
            mlp_ratio=config['mlp_ratio'],
            embedding_dropout=config['dropout_rate'],
            init_fn=config['init_fn'],
            layer_norm_type=config['layer_norm_type'],
            max_sequence_length=(config['max_item_seq_len'] + 1) *
            self.tokenizer.n_digit,
            n_heads=config['n_heads'],
            n_kv_heads=config['n_kv_heads'],
            n_layers=config['n_layers'],
            residual_dropout=config['dropout_rate'],
            rope=config['rope'],
            rope_theta=config['rope_theta'],
            weight_tying=config['weight_tying'],
            vocab_size=tokenizer.vocab_size,
            embedding_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            mask_token_id=self.mask_token_id)

        self.llada = LLaDAModelLM(lladaconfig)

        self.item_pos_emb = nn.Embedding(
            num_embeddings=config['max_item_seq_len'] + 1,
            embedding_dim=config['d_model'])
        init_weights(self.llada.model.config,
                     self.item_pos_emb,
                     type_of_module=ModuleType.emb)

        self.temperature = self.config['temperature']
        self.log("scale logits by temperature: {}".format(self.temperature))
        self.use_sid_trie_beam = _as_bool(
            config.get('use_sid_trie_beam', False))
        self.sid_trie_left_to_right = _as_bool(
            config.get('sid_trie_left_to_right', True))
        if self.use_sid_trie_beam:
            self.log(
                f"using SID trie constrained beam search: "
                f"left_to_right={self.sid_trie_left_to_right}")
        self.use_sid_anchor_condition = _as_bool(
            config.get('use_sid_anchor_condition', False))
        self.sid_anchor_scale = float(
            config.get('sid_anchor_scale', 0.03))
        self.sid_anchor_shuffle_condition = _as_bool(
            config.get('sid_anchor_shuffle_condition', False))
        self.sid_anchor_trainable_router = _as_bool(
            config.get('sid_anchor_trainable_router', False))
        self.sid_anchor_dropout = float(
            config.get('sid_anchor_dropout', 0.0))
        self.sid_anchor_router_aux_weight = float(
            config.get('sid_anchor_router_aux_weight', 0.0))
        self.sid_anchor_counterfactual_weight = float(
            config.get('sid_anchor_counterfactual_weight', 0.0))
        self.sid_anchor_counterfactual_margin = float(
            config.get('sid_anchor_counterfactual_margin', 0.02))
        self.sid_anchor_teacher_consistency_weight = float(
            config.get('sid_anchor_teacher_consistency_weight', 0.0))
        self.sid_anchor_item_contrastive_weight = float(
            config.get('sid_anchor_item_contrastive_weight', 0.0))
        self.sid_anchor_item_contrastive_temperature = float(
            config.get('sid_anchor_item_contrastive_temperature', 1.0))
        self.sid_anchor_digits = tuple(
            int(digit) - 1
            for digit in config.get('sid_anchor_digits', [2, 3, 4])
        )
        self.sid_anchor_router = None
        if self.use_sid_anchor_condition:
            router_path = config.get('sid_anchor_router_path')
            if not router_path:
                raise ValueError(
                    'use_sid_anchor_condition=True requires '
                    'sid_anchor_router_path')
            checkpoint = torch.load(
                router_path, map_location='cpu', weights_only=False)
            router_args = checkpoint['arguments']
            router_state = checkpoint['model_state_dict']
            item_sids = router_state['item_sids']
            if item_sids.shape[0] != self.dataset.n_items:
                raise ValueError(
                    f'SID anchor item count {item_sids.shape[0]} != '
                    f'{self.dataset.n_items}')
            self.sid_anchor_router = SIDAnchorRouter(
                num_items=self.dataset.n_items - 1,
                item_sids=item_sids,
                max_history=int(router_args['max_history']),
                dimension=int(router_args['dimension']),
                layers=int(router_args['layers']),
                heads=int(router_args['heads']),
                dropout=float(router_args['dropout']),
            )
            self.sid_anchor_router.load_state_dict(router_state)
            self.sid_anchor_teacher_router = None
            if (self.sid_anchor_trainable_router
                    and self.sid_anchor_teacher_consistency_weight > 0.0):
                self.sid_anchor_teacher_router = copy.deepcopy(
                    self.sid_anchor_router)
                self.sid_anchor_teacher_router.requires_grad_(False)
            self.sid_anchor_router.requires_grad_(
                self.sid_anchor_trainable_router)
            self.sid_anchor_scale_parameter = nn.Parameter(
                torch.tensor(self.sid_anchor_scale, dtype=torch.float32),
                requires_grad=self.sid_anchor_trainable_router,
            )
            if any(
                    digit < 0 or digit >= self.tokenizer.n_digit
                    for digit in self.sid_anchor_digits):
                raise ValueError(
                    f'Invalid sid_anchor_digits: '
                    f'{self.sid_anchor_digits}')
            self.log(
                f"using frozen SID anchor condition: "
                f"router={router_path}, "
                f"digits="
                f"{tuple(digit + 1 for digit in self.sid_anchor_digits)}, "
                f"scale={self.sid_anchor_scale}, "
                f"shuffle={self.sid_anchor_shuffle_condition}, "
                f"trainable_router={self.sid_anchor_trainable_router}, "
                f"dropout={self.sid_anchor_dropout}, "
                f"router_aux={self.sid_anchor_router_aux_weight}, "
                f"counterfactual={self.sid_anchor_counterfactual_weight}, "
                f"teacher_consistency="
                f"{self.sid_anchor_teacher_consistency_weight}, "
                f"item_contrastive="
                f"{self.sid_anchor_item_contrastive_weight}")

        self.use_dialogue_sid_condition = _as_bool(
            config.get('use_dialogue_sid_condition', False))
        self.dialogue_sid_condition_scale = float(
            config.get('dialogue_sid_condition_scale', 0.05))
        self.dialogue_sid_condition_path = config.get(
            'dialogue_sid_condition_path')
        if self.use_dialogue_sid_condition:
            if not self.dialogue_sid_condition_path:
                raise ValueError(
                    'use_dialogue_sid_condition=True requires '
                    'dialogue_sid_condition_path')
            dialogue_condition = np.load(self.dialogue_sid_condition_path)
            user_lookup = dialogue_condition['user_lookup'].astype(np.int64)
            local_sid_bias = dialogue_condition['local_sid_bias'].astype(
                np.float32)
            expected = (len(dialogue_condition['hard_user_ids']),
                        self.tokenizer.n_digit, 256)
            if local_sid_bias.shape != expected:
                raise ValueError(
                    f'dialogue SID bias {local_sid_bias.shape} != {expected}')
            self.register_buffer(
                'dialogue_sid_user_lookup',
                torch.from_numpy(user_lookup),
                persistent=False,
            )
            self.register_buffer(
                'dialogue_local_sid_bias',
                torch.from_numpy(local_sid_bias),
                persistent=False,
            )
            self.log(
                f"using external dialogue SID condition: "
                f"path={self.dialogue_sid_condition_path}, "
                f"users={local_sid_bias.shape[0]}, "
                f"scale={self.dialogue_sid_condition_scale}")

        self.use_user_codebook = _as_bool(config.get('use_user_codebook',
                                                     False))
        self.user_code_scale = float(config.get('user_code_scale', 1.0))
        self.user_code_temperature = float(
            config.get('user_code_temperature', 0.2))
        self.user_code_inject_mode = config.get('user_code_inject_mode',
                                                'all')
        self.use_user_code_logits_bias = _as_bool(
            config.get('use_user_code_logits_bias', False))
        self.user_code_logits_scale = float(
            config.get('user_code_logits_scale', 1.0))
        if self.use_user_codebook:
            self.num_user_codes = int(config.get('num_user_codes', 600))
            self.use_intent_codebook = _as_bool(
                config.get('use_intent_codebook', False))
            self.intent_codebook_path = config.get('intent_codebook_path')
            self.intent_codebook_weight = float(
                config.get('intent_codebook_weight', 0.5))
            self.trainable_intent_codebook = _as_bool(
                config.get('trainable_intent_codebook', False))
            self.user_codebook = nn.Parameter(
                torch.randn(self.num_user_codes, config['d_model']) * 0.02)
            gate_input_dim = config['d_model'] * (
                3 if self.use_intent_codebook else 2)
            self.user_code_gate = nn.Sequential(
                nn.Linear(gate_input_dim, config['d_model']),
                nn.SiLU(),
                nn.Dropout(config['dropout_rate']),
                nn.Linear(config['d_model'], config['d_model']),
            )
            nn.init.zeros_(self.user_code_gate[-1].weight)
            nn.init.zeros_(self.user_code_gate[-1].bias)
            self.user_code_norm = nn.LayerNorm(config['d_model'])
            if self.use_intent_codebook:
                if not self.intent_codebook_path:
                    raise ValueError(
                        'use_intent_codebook=True requires intent_codebook_path'
                    )
                intent_bank = torch.from_numpy(
                    np.load(self.intent_codebook_path).astype(np.float32))
                if intent_bank.ndim != 2:
                    raise ValueError(
                        f'intent codebook must be 2-D, got {intent_bank.shape}'
                    )
                if intent_bank.shape[0] != self.num_user_codes:
                    raise ValueError(
                        f'intent codebook rows {intent_bank.shape[0]} != num_user_codes {self.num_user_codes}'
                    )
                intent_dim = int(intent_bank.shape[1])
                if self.trainable_intent_codebook:
                    self.intent_codebook = nn.Parameter(intent_bank)
                else:
                    self.register_buffer('intent_codebook', intent_bank)
                if intent_dim == config['d_model']:
                    self.intent_proj = nn.Identity()
                else:
                    self.intent_proj = nn.Linear(intent_dim,
                                                 config['d_model'])
                self.intent_query = nn.Sequential(
                    nn.Linear(config['d_model'], config['d_model']),
                    nn.SiLU(),
                    nn.Linear(config['d_model'], config['d_model']),
                )
            if self.use_user_code_logits_bias:
                self.user_code_logits_bias = nn.Linear(
                    config['d_model'], tokenizer.vocab_size)
                nn.init.zeros_(self.user_code_logits_bias.weight)
                nn.init.zeros_(self.user_code_logits_bias.bias)
            self.log(
                f"using learnable user codebook: {self.num_user_codes} codes, "
                f"temperature={self.user_code_temperature}, scale={self.user_code_scale}, "
                f"inject_mode={self.user_code_inject_mode}, "
                f"logits_bias={self.use_user_code_logits_bias}, "
                f"intent_codebook={getattr(self, 'use_intent_codebook', False)}"
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.ignored_label, reduction='none')
        self.his_mask_w = self.config['his_mask_w']
        self.use_qwen_teacher_distill = _as_bool(
            config.get('use_qwen_teacher_distill', False))
        self.use_qwen_teacher_beam_prior = _as_bool(
            config.get('use_qwen_teacher_beam_prior', False))
        self.qwen_beam_prior_scale = float(
            config.get('qwen_beam_prior_scale', 0.0))
        self.qwen_beam_prior_alpha = float(
            config.get('qwen_beam_prior_alpha', 50.0))
        self.qwen_distill_weight = float(
            config.get('qwen_distill_weight', 0.0))
        self.qwen_distill_temperature = float(
            config.get('qwen_distill_temperature', 1.0))
        self.use_user_code_aux_recon = _as_bool(
            config.get('use_user_code_aux_recon', False))
        self.user_code_aux_weight = float(
            config.get('user_code_aux_weight', 0.0))
        self.user_code_aux_teacher_weight = float(
            config.get('user_code_aux_teacher_weight', 0.0))
        self.user_code_aux_temperature = float(
            config.get('user_code_aux_temperature', 1.0))
        self.user_code_aux_only = _as_bool(
            config.get('user_code_aux_only', False))
        if self.use_user_code_aux_recon and (
                not self.use_user_codebook
                or not self.use_user_code_logits_bias):
            raise ValueError(
                'use_user_code_aux_recon=True requires use_user_codebook=True and use_user_code_logits_bias=True'
            )
        needs_qwen_teacher = (
            self.use_qwen_teacher_distill or self.use_qwen_teacher_beam_prior
            or (self.use_user_code_aux_recon
                and self.user_code_aux_teacher_weight > 0.0))
        if needs_qwen_teacher:
            teacher_path = config.get('qwen_teacher_path')
            if not teacher_path:
                raise ValueError(
                    'Qwen teacher distillation/beam prior/aux recon requires qwen_teacher_path'
                )
            teacher = np.load(teacher_path)
            teacher_item_ids = torch.from_numpy(
                teacher['teacher_item_ids'].astype(np.int64))
            teacher_scores = torch.from_numpy(
                teacher['teacher_scores'].astype(np.float32))
            if teacher_item_ids.shape != teacher_scores.shape:
                raise ValueError(
                    f'teacher_item_ids shape {teacher_item_ids.shape} != teacher_scores shape {teacher_scores.shape}'
                )
            self.register_buffer('qwen_teacher_item_ids', teacher_item_ids)
            self.register_buffer('qwen_teacher_scores', teacher_scores)
            self.log(
                f"using Qwen teacher signals: path={teacher_path}, "
                f"shape={tuple(teacher_item_ids.shape)}, "
                f"distill={self.use_qwen_teacher_distill}, "
                f"distill_weight={self.qwen_distill_weight}, "
                f"distill_temperature={self.qwen_distill_temperature}, "
                f"beam_prior={self.use_qwen_teacher_beam_prior}, "
                f"beam_prior_scale={self.qwen_beam_prior_scale}, "
                f"beam_prior_alpha={self.qwen_beam_prior_alpha}")
        if self.use_user_code_aux_recon:
            self.log(
                f"using user-code auxiliary SID reconstruction: "
                f"weight={self.user_code_aux_weight}, "
                f"teacher_weight={self.user_code_aux_teacher_weight}, "
                f"temperature={self.user_code_aux_temperature}, "
                f"aux_only={self.user_code_aux_only}")
        self.use_user_code_mask_recon = _as_bool(
            config.get('use_user_code_mask_recon', False))
        self.user_code_mask_weight = float(
            config.get('user_code_mask_weight', 0.0))
        self.user_code_mask_ratio = float(
            config.get('user_code_mask_ratio', 0.35))
        self.user_code_mask_temperature = float(
            config.get('user_code_mask_temperature', 0.2))
        self.user_code_mask_context_weight = float(
            config.get('user_code_mask_context_weight', 0.1))
        self.user_code_mask_only = _as_bool(
            config.get('user_code_mask_only', False))
        if self.use_user_code_mask_recon and not self.use_user_codebook:
            raise ValueError(
                'use_user_code_mask_recon=True requires use_user_codebook=True'
            )
        if self.use_user_code_mask_recon:
            self.log(
                f"using user-code-space mask reconstruction: "
                f"weight={self.user_code_mask_weight}, "
                f"ratio={self.user_code_mask_ratio}, "
                f"temperature={self.user_code_mask_temperature}, "
                f"context_weight={self.user_code_mask_context_weight}, "
                f"mask_only={self.user_code_mask_only}")

        self.use_qwen_reasoning_adapter = _as_bool(
            config.get('use_qwen_reasoning_adapter', False))
        self.qwen_reasoning_scale = float(
            config.get('qwen_reasoning_scale', 1.0))
        self.qwen_reasoning_inject_mode = config.get(
            'qwen_reasoning_inject_mode', 'target_mask')
        self.user_condition_guidance_scale = float(
            config.get('user_condition_guidance_scale', 0.0))
        self.user_condition_dropout = float(
            config.get('user_condition_dropout', 0.0))
        self.target_full_mask_prob = float(
            config.get('target_full_mask_prob', 0.0))
        if self.use_qwen_reasoning_adapter:
            reasoning_path = config.get('qwen_reasoning_path')
            if not reasoning_path:
                raise ValueError(
                    'use_qwen_reasoning_adapter=True requires qwen_reasoning_path'
                )
            reasoning_npz = np.load(reasoning_path)
            reasoning_code_emb = torch.from_numpy(
                reasoning_npz['code_reasoning_emb'].astype(np.float32))
            reasoning_user_code_ids = torch.from_numpy(
                reasoning_npz['user_code_ids'].astype(np.int64))
            if reasoning_code_emb.ndim != 2:
                raise ValueError(
                    f'code_reasoning_emb must be 2-D, got {reasoning_code_emb.shape}'
                )
            self.register_buffer('qwen_reasoning_code_emb',
                                 reasoning_code_emb)
            self.register_buffer('qwen_reasoning_user_code_ids',
                                 reasoning_user_code_ids)
            reasoning_dim = int(reasoning_code_emb.shape[1])
            self.qwen_reasoning_adapter = nn.Sequential(
                nn.Linear(reasoning_dim, config['d_model']),
                nn.SiLU(),
                nn.Dropout(config['dropout_rate']),
                nn.Linear(config['d_model'], config['d_model']),
            )
            nn.init.zeros_(self.qwen_reasoning_adapter[-1].weight)
            nn.init.zeros_(self.qwen_reasoning_adapter[-1].bias)
            self.qwen_reasoning_norm = nn.LayerNorm(config['d_model'])
            self.log(
                f"using Qwen reasoning adapter: path={reasoning_path}, "
                f"code_shape={tuple(reasoning_code_emb.shape)}, "
                f"scale={self.qwen_reasoning_scale}, "
                f"inject_mode={self.qwen_reasoning_inject_mode}")
        if self.user_condition_dropout > 0.0:
            self.log(
                f"using user condition dropout during training: "
                f"p={self.user_condition_dropout}")
        if self.target_full_mask_prob > 0.0:
            self.log(
                f"using target SID full-mask reconstruction: "
                f"prob={self.target_full_mask_prob}")

        self.use_rec_controlnet = _as_bool(
            config.get('use_rec_controlnet', False))
        self.rec_control_scale = float(config.get('rec_control_scale', 1.0))
        self.rec_control_inject_mode = config.get('rec_control_inject_mode',
                                                  'target_mask')
        self.rec_control_disable_input_injection = _as_bool(
            config.get('rec_control_disable_input_injection', True))
        self.rec_control_use_user_code = _as_bool(
            config.get('rec_control_use_user_code', True))
        self.rec_control_use_qwen = _as_bool(
            config.get('rec_control_use_qwen', True))
        self.use_rec_control_pretrain = _as_bool(
            config.get('use_rec_control_pretrain', False))
        self.rec_control_pretrain_weight = float(
            config.get('rec_control_pretrain_weight', 0.0))
        self.rec_control_pretrain_qwen_weight = float(
            config.get('rec_control_pretrain_qwen_weight', 0.0))
        self.rec_control_pretrain_only = _as_bool(
            config.get('rec_control_pretrain_only', False))
        self.rec_control_align_weight = float(
            config.get('rec_control_align_weight', 0.0))
        self.rec_control_align_temperature = float(
            config.get('rec_control_align_temperature', 0.1))
        self.use_rec_control_item_rank = _as_bool(
            config.get('use_rec_control_item_rank', False))
        self.rec_control_item_rank_weight = float(
            config.get('rec_control_item_rank_weight', 0.0))
        self.rec_control_item_rank_teacher_weight = float(
            config.get('rec_control_item_rank_teacher_weight', 0.0))
        self.rec_control_item_rank_temperature = float(
            config.get('rec_control_item_rank_temperature', 0.1))
        self.rec_control_item_rank_use_sampled = _as_bool(
            config.get('rec_control_item_rank_use_sampled', False))
        self.rec_control_item_rank_num_teacher = int(
            config.get('rec_control_item_rank_num_teacher', 20))
        self.rec_control_item_rank_num_random = int(
            config.get('rec_control_item_rank_num_random', 64))
        self.rec_control_item_rank_label_weight = float(
            config.get('rec_control_item_rank_label_weight', 1.0))
        self.rec_control_item_rank_inbatch_weight = float(
            config.get('rec_control_item_rank_inbatch_weight', 0.0))
        self.use_rec_control_sid_token = _as_bool(
            config.get('use_rec_control_sid_token', False))
        self.rec_control_sid_token_weight = float(
            config.get('rec_control_sid_token_weight', 0.0))
        self.rec_control_sid_token_temperature = float(
            config.get('rec_control_sid_token_temperature', 1.0))
        self.rec_control_sid_token_prefix_decay = float(
            config.get('rec_control_sid_token_prefix_decay', 1.0))
        self.rec_control_sid_token_teacher_weight = float(
            config.get('rec_control_sid_token_teacher_weight', 0.0))
        if self.use_rec_controlnet:
            if (not self.use_user_codebook
                    and not self.use_qwen_reasoning_adapter):
                raise ValueError(
                    'use_rec_controlnet=True requires user codebook or Qwen reasoning adapter'
                )
            self.rec_control_norm = nn.LayerNorm(config['d_model'])
            self.rec_control_adapters = nn.ModuleList()
            for _ in range(config['n_layers']):
                adapter = nn.Sequential(
                    nn.LayerNorm(config['d_model']),
                    nn.Linear(config['d_model'], config['d_model']),
                    nn.SiLU(),
                    nn.Dropout(config['dropout_rate']),
                    nn.Linear(config['d_model'], config['d_model']),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.rec_control_adapters.append(adapter)
            if (self.use_rec_control_pretrain
                    and self.rec_control_align_weight > 0.0):
                self.rec_control_align_proj = nn.Linear(
                    config['d_model'], config['d_model'])
            if self.use_rec_control_item_rank:
                self.rec_control_item_rank_proj = nn.Linear(
                    config['d_model'], config['d_model'])
            if self.use_rec_control_sid_token:
                self.rec_control_sid_token_proj = nn.Linear(
                    config['d_model'],
                    self.tokenizer.n_digit * self.tokenizer.vocab_size)
            self.log(
                f"using Rec-ControlNet condition branch: "
                f"scale={self.rec_control_scale}, "
                f"inject_mode={self.rec_control_inject_mode}, "
                f"disable_input_injection={self.rec_control_disable_input_injection}, "
                f"use_user_code={self.rec_control_use_user_code}, "
                f"use_qwen={self.rec_control_use_qwen}")
        elif self.use_rec_control_pretrain:
            raise ValueError(
                'use_rec_control_pretrain=True requires use_rec_controlnet=True'
            )
        if self.use_rec_control_pretrain:
            self.log(
                f"using Rec-ControlNet pretraining: "
                f"weight={self.rec_control_pretrain_weight}, "
                f"qwen_weight={self.rec_control_pretrain_qwen_weight}, "
                f"only={self.rec_control_pretrain_only}, "
                f"align_weight={self.rec_control_align_weight}, "
                f"align_temperature={self.rec_control_align_temperature}")
        if self.use_rec_control_item_rank:
            if not self.use_rec_controlnet:
                raise ValueError(
                    'use_rec_control_item_rank=True requires use_rec_controlnet=True'
                )
            self.log(
                f"using Rec-ControlNet item-level ranking loss: "
                f"weight={self.rec_control_item_rank_weight}, "
                f"teacher_weight={self.rec_control_item_rank_teacher_weight}, "
                f"temperature={self.rec_control_item_rank_temperature}, "
                f"use_sampled={self.rec_control_item_rank_use_sampled}, "
                f"num_teacher={self.rec_control_item_rank_num_teacher}, "
                f"num_random={self.rec_control_item_rank_num_random}, "
                f"label_weight={self.rec_control_item_rank_label_weight}, "
                f"inbatch_weight={self.rec_control_item_rank_inbatch_weight}")
        if self.use_rec_control_sid_token:
            if not self.use_rec_controlnet:
                raise ValueError(
                    'use_rec_control_sid_token=True requires use_rec_controlnet=True'
                )
            self.log(
                f"using Rec-ControlNet SID-token condition loss: "
                f"weight={self.rec_control_sid_token_weight}, "
                f"temperature={self.rec_control_sid_token_temperature}, "
                f"prefix_decay={self.rec_control_sid_token_prefix_decay}, "
                f"teacher_weight={self.rec_control_sid_token_teacher_weight}")

        self.use_semantic_bridge = _as_bool(
            config.get('use_semantic_bridge', False))
        self.semantic_bridge_weight = float(
            config.get('semantic_bridge_weight', 0.0))
        self.semantic_bridge_contrastive_weight = float(
            config.get('semantic_bridge_contrastive_weight', 0.1))
        self.semantic_bridge_temperature = float(
            config.get('semantic_bridge_temperature', 0.1))
        self.semantic_bridge_scale = float(
            config.get('semantic_bridge_scale', 1.0))
        self.semantic_bridge_inject_mode = config.get(
            'semantic_bridge_inject_mode', 'target_mask')
        self.semantic_bridge_shuffle_condition = _as_bool(
            config.get('semantic_bridge_shuffle_condition', False))
        if self.use_semantic_bridge:
            semantic_path = config.get('semantic_bridge_item_path')
            if not semantic_path:
                raise ValueError(
                    'use_semantic_bridge=True requires semantic_bridge_item_path'
                )
            raw_semantics = np.fromfile(semantic_path, dtype=np.float32)
            num_catalog_items = self.dataset.n_items - 1
            if raw_semantics.size % num_catalog_items != 0:
                raise ValueError(
                    f'item semantic file size {raw_semantics.size} is not '
                    f'divisible by {num_catalog_items} catalog items')
            semantic_dim = raw_semantics.size // num_catalog_items
            raw_semantics = raw_semantics.reshape(
                num_catalog_items, semantic_dim)
            raw_semantics = raw_semantics / np.maximum(
                np.linalg.norm(raw_semantics, axis=1, keepdims=True), 1e-8)
            item_semantics = np.concatenate(
                (np.zeros((1, semantic_dim), dtype=np.float32),
                 raw_semantics),
                axis=0,
            )
            item_semantics = torch.from_numpy(item_semantics)
            self.register_buffer('semantic_bridge_item_embeddings',
                                 item_semantics)
            token_bank = self._build_semantic_token_bank(
                item_semantics, tokenizer.vocab_size)
            self.register_buffer('semantic_bridge_token_bank', token_bank)
            self.semantic_bridge_context = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, config['d_model'] * 2),
                nn.SiLU(),
                nn.Linear(config['d_model'] * 2, config['d_model']),
                nn.LayerNorm(config['d_model']),
            )
            self.semantic_bridge_adapters = nn.ModuleList()
            for _ in range(config['n_layers']):
                adapter = nn.Sequential(
                    nn.LayerNorm(config['d_model']),
                    nn.Linear(config['d_model'], config['d_model']),
                    nn.SiLU(),
                    nn.Dropout(config['dropout_rate']),
                    nn.Linear(config['d_model'], config['d_model']),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.semantic_bridge_adapters.append(adapter)
            self.semantic_bridge_predictor = nn.Sequential(
                nn.LayerNorm(config['d_model']),
                nn.Linear(config['d_model'], config['d_model'] * 2),
                nn.SiLU(),
                nn.Linear(config['d_model'] * 2, semantic_dim),
            )
            self.log(
                f"using Semantic-Bridge diffusion alignment: "
                f"path={semantic_path}, semantic_dim={semantic_dim}, "
                f"loss_weight={self.semantic_bridge_weight}, "
                f"contrastive_weight={self.semantic_bridge_contrastive_weight}, "
                f"scale={self.semantic_bridge_scale}, "
                f"inject_mode={self.semantic_bridge_inject_mode}, "
                f"shuffle={self.semantic_bridge_shuffle_condition}")

        self.use_multi_intent_memory = _as_bool(
            config.get('use_multi_intent_memory', False))
        self.multi_intent_recon_weight = float(
            config.get('multi_intent_recon_weight', 0.0))
        self.multi_intent_contrastive_weight = float(
            config.get('multi_intent_contrastive_weight', 0.1))
        self.multi_intent_assignment_weight = float(
            config.get('multi_intent_assignment_weight', 0.5))
        self.multi_intent_diversity_weight = float(
            config.get('multi_intent_diversity_weight', 0.05))
        self.multi_intent_temperature = float(
            config.get('multi_intent_temperature', 0.1))
        self.multi_intent_scale = float(
            config.get('multi_intent_scale', 1.0))
        self.multi_intent_inject_mode = config.get(
            'multi_intent_inject_mode', 'target_mask')
        self.multi_intent_shuffle_condition = _as_bool(
            config.get('multi_intent_shuffle_condition', False))
        if self.use_multi_intent_memory:
            semantic_path = config.get('multi_intent_item_path')
            if not semantic_path:
                raise ValueError(
                    'use_multi_intent_memory=True requires '
                    'multi_intent_item_path')
            raw_semantics = np.fromfile(semantic_path, dtype=np.float32)
            num_catalog_items = self.dataset.n_items - 1
            if raw_semantics.size % num_catalog_items != 0:
                raise ValueError(
                    f'item semantic file size {raw_semantics.size} is not '
                    f'divisible by {num_catalog_items} catalog items')
            semantic_dim = raw_semantics.size // num_catalog_items
            raw_semantics = raw_semantics.reshape(
                num_catalog_items, semantic_dim)
            raw_semantics = raw_semantics / np.maximum(
                np.linalg.norm(raw_semantics, axis=1, keepdims=True), 1e-8)
            item_semantics = np.concatenate(
                (np.zeros((1, semantic_dim), dtype=np.float32),
                 raw_semantics),
                axis=0,
            )
            self.register_buffer(
                'multi_intent_item_embeddings',
                torch.from_numpy(item_semantics),
            )

            d_model = config['d_model']
            num_slots = int(config.get('multi_intent_num_slots', 4))
            num_heads = int(config.get('multi_intent_num_heads',
                                       config['n_heads']))
            if d_model % num_heads != 0:
                raise ValueError(
                    f'd_model={d_model} must be divisible by '
                    f'multi_intent_num_heads={num_heads}')
            self.multi_intent_num_slots = num_slots
            self.multi_intent_item_proj = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )
            self.multi_intent_history_pos = nn.Embedding(
                config['max_item_seq_len'], d_model)
            self.multi_intent_queries = nn.Parameter(
                torch.randn(num_slots, d_model) * 0.02)
            self.multi_intent_slot_attention = nn.MultiheadAttention(
                d_model, num_heads, dropout=config['dropout_rate'],
                batch_first=True)
            self.multi_intent_slot_norm = nn.LayerNorm(d_model)
            self.multi_intent_gate = nn.Linear(d_model, 1)
            self.multi_intent_cross_attention = nn.ModuleList([
                nn.MultiheadAttention(
                    d_model, num_heads, dropout=config['dropout_rate'],
                    batch_first=True)
                for _ in range(config['n_layers'])
            ])
            self.multi_intent_residual_norm = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(config['n_layers'])
            ])
            initial_gate = float(
                config.get('multi_intent_initial_residual_gate', 0.1))
            initial_gate = min(max(initial_gate, 1e-4), 1.0 - 1e-4)
            gate_logit = np.log(initial_gate / (1.0 - initial_gate))
            self.multi_intent_layer_gates = nn.Parameter(
                torch.full((config['n_layers'], ), float(gate_logit)))
            self.multi_intent_semantic_decoder = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, semantic_dim),
            )
            self.log(
                f"using Multi-Intent Semantic Memory diffusion: "
                f"path={semantic_path}, semantic_dim={semantic_dim}, "
                f"slots={num_slots}, heads={num_heads}, "
                f"recon_weight={self.multi_intent_recon_weight}, "
                f"scale={self.multi_intent_scale}, "
                f"inject_mode={self.multi_intent_inject_mode}, "
                f"shuffle={self.multi_intent_shuffle_condition}")

        self.use_dialogue_multi_intent = _as_bool(
            config.get('use_dialogue_multi_intent', False))
        self.dialogue_multi_intent_recon_weight = float(
            config.get('dialogue_multi_intent_recon_weight', 0.0))
        self.dialogue_multi_intent_assignment_weight = float(
            config.get('dialogue_multi_intent_assignment_weight', 0.5))
        self.dialogue_multi_intent_contrastive_weight = float(
            config.get('dialogue_multi_intent_contrastive_weight', 0.1))
        self.dialogue_multi_intent_diversity_weight = float(
            config.get('dialogue_multi_intent_diversity_weight', 0.05))
        self.dialogue_multi_intent_temperature = float(
            config.get('dialogue_multi_intent_temperature', 0.1))
        self.dialogue_multi_intent_scale = float(
            config.get('dialogue_multi_intent_scale', 1.0))
        self.dialogue_multi_intent_inject_mode = config.get(
            'dialogue_multi_intent_inject_mode', 'target_mask')
        self.dialogue_multi_intent_shuffle_condition = _as_bool(
            config.get('dialogue_multi_intent_shuffle_condition', False))
        self.dialogue_multi_intent_active_slot = int(
            config.get('dialogue_multi_intent_active_slot', -1))
        self.dialogue_multi_intent_shuffle_offset = int(
            config.get('dialogue_multi_intent_shuffle_offset', 7919))
        self.use_dialogue_multi_intent_sid_guidance = _as_bool(
            config.get('use_dialogue_multi_intent_sid_guidance', False))
        self.dialogue_multi_intent_sid_guidance_scale = float(
            config.get('dialogue_multi_intent_sid_guidance_scale', 0.0))
        if self.use_dialogue_multi_intent:
            intent_path = config.get('dialogue_multi_intent_path')
            if not intent_path:
                raise ValueError(
                    'use_dialogue_multi_intent=True requires '
                    'dialogue_multi_intent_path')
            intent_data = np.load(intent_path)
            intent_embeddings = intent_data['intent_embeddings'].astype(
                np.float32)
            intent_probabilities = intent_data[
                'intent_probabilities'].astype(np.float32)
            item_embeddings = intent_data['item_embeddings'].astype(
                np.float32)
            if intent_embeddings.ndim != 3:
                raise ValueError(
                    f'intent_embeddings must be 3-D, got '
                    f'{intent_embeddings.shape}')
            if intent_probabilities.shape != intent_embeddings.shape[:2]:
                raise ValueError(
                    f'intent probabilities {intent_probabilities.shape} do '
                    f'not match intents {intent_embeddings.shape[:2]}')
            if item_embeddings.shape[0] != self.dataset.n_items:
                raise ValueError(
                    f'item embeddings contain {item_embeddings.shape[0]} '
                    f'rows, expected {self.dataset.n_items}')
            if item_embeddings.shape[1] != intent_embeddings.shape[2]:
                raise ValueError(
                    'dialogue intent and item embedding dimensions differ')
            self.register_buffer(
                'dialogue_multi_intent_embeddings',
                torch.from_numpy(intent_embeddings), persistent=False)
            self.register_buffer(
                'dialogue_multi_intent_probabilities',
                torch.from_numpy(intent_probabilities), persistent=False)
            self.register_buffer(
                'dialogue_multi_intent_item_embeddings',
                torch.from_numpy(item_embeddings), persistent=False)
            if self.use_dialogue_multi_intent_sid_guidance:
                guidance_path = config.get(
                    'dialogue_multi_intent_sid_guidance_path')
                if not guidance_path:
                    raise ValueError(
                        'Intent SID guidance requires '
                        'dialogue_multi_intent_sid_guidance_path')
                guidance_data = np.load(guidance_path)
                intent_sid_bias = guidance_data['intent_sid_bias'].astype(
                    np.float32)
                expected_prefix = (
                    intent_embeddings.shape[0], intent_embeddings.shape[1],
                    self.tokenizer.n_digit)
                if intent_sid_bias.shape[:3] != expected_prefix:
                    raise ValueError(
                        f'intent SID bias shape {intent_sid_bias.shape} does '
                        f'not start with {expected_prefix}')
                self.register_buffer(
                    'dialogue_multi_intent_sid_bias',
                    torch.from_numpy(intent_sid_bias), persistent=False)

            d_model = config['d_model']
            semantic_dim = intent_embeddings.shape[-1]
            num_slots = intent_embeddings.shape[1]
            num_heads = int(config.get('multi_intent_num_heads',
                                       config['n_heads']))
            self.dialogue_multi_intent_num_slots = num_slots
            self.dialogue_multi_intent_projection = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )
            self.dialogue_multi_intent_slot_positions = nn.Parameter(
                torch.randn(num_slots, d_model) * 0.02)
            self.dialogue_multi_intent_cross_attention = nn.ModuleList([
                nn.MultiheadAttention(
                    d_model, num_heads, dropout=config['dropout_rate'],
                    batch_first=True)
                for _ in range(config['n_layers'])
            ])
            self.dialogue_multi_intent_residual_norm = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(config['n_layers'])
            ])
            initial_gate = float(config.get(
                'dialogue_multi_intent_initial_residual_gate', 0.05))
            initial_gate = min(max(initial_gate, 1e-4), 1.0 - 1e-4)
            gate_logit = np.log(initial_gate / (1.0 - initial_gate))
            self.dialogue_multi_intent_layer_gates = nn.Parameter(
                torch.full((config['n_layers'], ), float(gate_logit)))
            self.dialogue_multi_intent_decoder = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, semantic_dim),
            )
            self.log(
                f"using dialogue-derived multi-intent diffusion: "
                f"path={intent_path}, users={intent_embeddings.shape[0]}, "
                f"slots={num_slots}, semantic_dim={semantic_dim}, "
                f"recon_weight={self.dialogue_multi_intent_recon_weight}, "
                f"scale={self.dialogue_multi_intent_scale}, "
                f"shuffle={self.dialogue_multi_intent_shuffle_condition}, "
                f"active_slot={self.dialogue_multi_intent_active_slot}, "
                f"sid_guidance="
                f"{self.use_dialogue_multi_intent_sid_guidance}, "
                f"sid_guidance_scale="
                f"{self.dialogue_multi_intent_sid_guidance_scale}")

        self.use_probabilistic_intent_memory = _as_bool(
            config.get('use_probabilistic_intent_memory', False))
        self.prob_intent_scale = float(
            config.get('prob_intent_scale', 1.0))
        self.prob_intent_inject_mode = config.get(
            'prob_intent_inject_mode', 'target_mask')
        self.prob_intent_shuffle_condition = _as_bool(
            config.get('prob_intent_shuffle_condition', False))
        self.prob_intent_recency_decay = float(
            config.get('prob_intent_recency_decay', 0.2))
        self.prob_intent_use_sid_prior = _as_bool(
            config.get('prob_intent_use_sid_prior', True))
        self.prob_intent_use_full_distribution_sid_prior = _as_bool(
            config.get(
                'prob_intent_use_full_distribution_sid_prior',
                False))
        self.prob_intent_use_hierarchy_sid_prior = _as_bool(
            config.get('prob_intent_use_hierarchy_sid_prior', False))
        self.prob_intent_use_catalog_decoder = _as_bool(
            config.get('prob_intent_use_catalog_decoder', False))
        self.prob_intent_trainable_router = _as_bool(
            config.get('prob_intent_trainable_router', False))
        self.prob_intent_router_aux_weight = float(
            config.get('prob_intent_router_aux_weight', 0.0))
        self.prob_intent_counterfactual_weight = float(
            config.get('prob_intent_counterfactual_weight', 0.0))
        self.prob_intent_counterfactual_margin = float(
            config.get('prob_intent_counterfactual_margin', 0.05))
        if self.use_probabilistic_intent_memory:
            item_category_path = config.get('prob_intent_item_category_path')
            category_token_path = config.get('prob_intent_category_token_path')
            checkpoint_path = config.get('prob_intent_checkpoint_path')
            if not item_category_path or not category_token_path:
                raise ValueError(
                    'use_probabilistic_intent_memory=True requires item '
                    'category and category token paths')

            item_categories = np.load(item_category_path).astype(np.int64)
            category_tokens = np.load(category_token_path).astype(np.float32)
            if item_categories.shape != (self.dataset.n_items, ):
                raise ValueError(
                    f'item categories {item_categories.shape} != '
                    f'({self.dataset.n_items},)')
            if category_tokens.ndim != 3:
                raise ValueError(
                    f'category tokens must be 3-D, got '
                    f'{category_tokens.shape}')
            self.register_buffer(
                'prob_intent_item_categories',
                torch.from_numpy(item_categories),
            )
            self.register_buffer(
                'prob_intent_category_tokens',
                torch.from_numpy(category_tokens),
            )
            transition_path = config.get('prob_intent_transition_path')
            if transition_path:
                transition = np.load(transition_path).astype(np.float32)
                expected = (
                    category_tokens.shape[0], category_tokens.shape[0])
                if transition.shape != expected:
                    raise ValueError(
                        f'category transition {transition.shape} != '
                        f'{expected}')
                self.register_buffer(
                    'prob_intent_category_transition',
                    torch.from_numpy(transition),
                )
                if self.prob_intent_trainable_router:
                    self.prob_intent_transition_logits = nn.Parameter(
                        torch.from_numpy(
                            np.log(transition.clip(min=1e-8))))
            else:
                self.prob_intent_category_transition = None
                if self.prob_intent_trainable_router:
                    raise ValueError(
                        'prob_intent_trainable_router=True requires a '
                        'category transition asset')

            learned_router_path = config.get(
                'prob_intent_learned_router_path')
            leaf_to_hierarchy_path = config.get(
                'prob_intent_leaf_to_hierarchy_path')
            self.prob_intent_learned_router = None
            if learned_router_path:
                router_checkpoint = torch.load(
                    learned_router_path,
                    map_location='cpu',
                    weights_only=False,
                )
                router_args = router_checkpoint['arguments']
                router_state = router_checkpoint['model_state_dict']
                hierarchy_sizes = [
                    int(router_state[
                        f'hierarchy_heads.{level}.weight'].shape[0])
                    for level in range(3)
                ]
                self.prob_intent_learned_router = \
                    HierarchicalCategoryRouter(
                        num_items=self.dataset.n_items - 1,
                        item_categories=torch.from_numpy(item_categories),
                        category_count=category_tokens.shape[0],
                        hierarchy_sizes=hierarchy_sizes,
                        max_history=int(router_args['max_history']),
                        dimension=int(router_args['dimension']),
                        layers=int(router_args['layers']),
                        heads=int(router_args['heads']),
                        dropout=float(router_args['dropout']),
                    )
                self.prob_intent_learned_router.load_state_dict(
                    router_state)
                self.prob_intent_learned_router.requires_grad_(False)

            d_model = config['d_model']
            text_dim = int(category_tokens.shape[-1])
            num_slots = int(config.get('prob_intent_num_slots', 4))
            num_heads = int(config.get(
                'prob_intent_num_heads', config['n_heads']))
            if d_model % num_heads != 0:
                raise ValueError(
                    f'd_model={d_model} must be divisible by '
                    f'prob_intent_num_heads={num_heads}')
            if num_slots > category_tokens.shape[1]:
                raise ValueError(
                    f'prob_intent_num_slots={num_slots} exceeds available '
                    f'type embeddings {category_tokens.shape[1]}')
            self.prob_intent_num_slots = num_slots
            self.prob_intent_text_norm = nn.LayerNorm(text_dim)
            self.prob_intent_text_projection = nn.Sequential(
                nn.Linear(text_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.prob_intent_type_embedding = nn.Parameter(
                torch.zeros(num_slots, d_model))
            self.prob_intent_query_norm = nn.LayerNorm(d_model)
            self.prob_intent_cross_attention = nn.ModuleList([
                nn.MultiheadAttention(
                    d_model,
                    num_heads,
                    dropout=config['dropout_rate'],
                    batch_first=True,
                ) for _ in range(config['n_layers'])
            ])
            self.prob_intent_residual_norm = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(config['n_layers'])
            ])
            initial_gate = float(
                config.get('prob_intent_initial_residual_gate', 0.05))
            initial_gate = min(max(initial_gate, 1e-4), 1.0 - 1e-4)
            gate_logit = np.log(initial_gate / (1.0 - initial_gate))
            self.prob_intent_layer_gates = nn.Parameter(
                torch.full((config['n_layers'], ), float(gate_logit)))
            sid_prior_initial = float(
                config.get('prob_intent_sid_prior_initial_scale', 0.05))
            self.prob_intent_sid_prior_scales = nn.Parameter(
                torch.full(
                    (self.tokenizer.n_digit, ), sid_prior_initial))
            catalog_initial = float(
                config.get('prob_intent_catalog_initial_scale', 0.1))
            self.prob_intent_catalog_scales = nn.Parameter(
                torch.full(
                    (self.tokenizer.n_digit, ), catalog_initial))
            if self.prob_intent_use_sid_prior:
                item_tokens = self.item_id2tokens.detach().cpu().numpy()
                category_count = category_tokens.shape[0]
                digit_count = self.tokenizer.n_digit
                vocab_size = self.tokenizer.vocab_size
                smoothing = 0.05
                counts = np.full(
                    (category_count, digit_count, vocab_size),
                    smoothing,
                    dtype=np.float64,
                )
                global_counts = np.full(
                    (digit_count, vocab_size), smoothing, dtype=np.float64)
                valid_categories = item_categories[1:]
                valid_tokens = item_tokens[1:]
                for depth in range(digit_count):
                    np.add.at(
                        counts[:, depth, :],
                        (valid_categories, valid_tokens[:, depth]),
                        1.0,
                    )
                    np.add.at(
                        global_counts[depth],
                        valid_tokens[:, depth],
                        1.0,
                    )
                category_probs = counts / counts.sum(
                    axis=-1, keepdims=True)
                global_probs = global_counts / global_counts.sum(
                    axis=-1, keepdims=True)
                log_likelihood_ratio = np.log(
                    category_probs / global_probs[None])
                log_likelihood_ratio = np.clip(
                    log_likelihood_ratio, -3.0, 3.0)
                self.register_buffer(
                    'prob_intent_category_sid_llr',
                    torch.from_numpy(
                        log_likelihood_ratio.astype(np.float32)),
                )
                self.prob_intent_hierarchy_depths = 0
                if leaf_to_hierarchy_path:
                    leaf_to_hierarchy = np.load(
                        leaf_to_hierarchy_path).astype(np.int64)
                    expected = (category_count, 3)
                    if leaf_to_hierarchy.shape != expected:
                        raise ValueError(
                            f'leaf hierarchy {leaf_to_hierarchy.shape} != '
                            f'{expected}')
                    self.register_buffer(
                        'prob_intent_leaf_to_hierarchy',
                        torch.from_numpy(leaf_to_hierarchy),
                    )
                    valid_hierarchy = leaf_to_hierarchy[valid_categories]
                    self.prob_intent_hierarchy_depths = (
                        leaf_to_hierarchy.shape[1])
                    for hierarchy_depth in range(
                            self.prob_intent_hierarchy_depths):
                        node_count = int(
                            leaf_to_hierarchy[:, hierarchy_depth].max()) + 1
                        hierarchy_counts = np.full(
                            (node_count, digit_count, vocab_size),
                            smoothing,
                            dtype=np.float64,
                        )
                        for digit_depth in range(digit_count):
                            np.add.at(
                                hierarchy_counts[:, digit_depth, :],
                                (
                                    valid_hierarchy[:, hierarchy_depth],
                                    valid_tokens[:, digit_depth],
                                ),
                                1.0,
                            )
                        hierarchy_probs = (
                            hierarchy_counts
                            / hierarchy_counts.sum(axis=-1, keepdims=True)
                        )
                        hierarchy_llr = np.log(
                            hierarchy_probs / global_probs[None])
                        hierarchy_llr = np.clip(
                            hierarchy_llr, -3.0, 3.0)
                        self.register_buffer(
                            f'prob_intent_hierarchy_sid_llr_'
                            f'{hierarchy_depth}',
                            torch.from_numpy(
                                hierarchy_llr.astype(np.float32)),
                        )
                elif self.prob_intent_use_hierarchy_sid_prior:
                    raise ValueError(
                        'prob_intent_use_hierarchy_sid_prior=True requires '
                        'prob_intent_leaf_to_hierarchy_path')
            self.prob_intent_source_checkpoint = checkpoint_path
            if checkpoint_path:
                checkpoint = torch.load(
                    checkpoint_path, map_location='cpu', weights_only=False)
                source = checkpoint.get('model_state_dict', checkpoint)
                self.prob_intent_text_norm.load_state_dict({
                    'weight': source['text_norm.weight'],
                    'bias': source['text_norm.bias'],
                })
                self.prob_intent_text_projection.load_state_dict({
                    key.removeprefix('text_projection.'): value
                    for key, value in source.items()
                    if key.startswith('text_projection.')
                })
                with torch.no_grad():
                    self.prob_intent_type_embedding.copy_(
                        source['text_type_embedding'][:num_slots])
                attention_state = {
                    key.removeprefix('text_cross_attention.'): value
                    for key, value in source.items()
                    if key.startswith('text_cross_attention.')
                }
                for attention in self.prob_intent_cross_attention:
                    attention.load_state_dict(attention_state)
            self.log(
                f"using history-derived probabilistic intent memory: "
                f"item_categories={item_category_path}, "
                f"category_tokens={category_token_path}, "
                f"checkpoint={checkpoint_path}, slots={num_slots}, "
                f"transition={transition_path}, "
                f"learned_router={learned_router_path}, "
                f"trainable_router={self.prob_intent_trainable_router}, "
                f"router_aux_weight={self.prob_intent_router_aux_weight}, "
                f"sid_prior={self.prob_intent_use_sid_prior}, "
                f"full_sid_prior="
                f"{self.prob_intent_use_full_distribution_sid_prior}, "
                f"hierarchy_sid_prior="
                f"{self.prob_intent_use_hierarchy_sid_prior}, "
                f"leaf_to_hierarchy={leaf_to_hierarchy_path}, "
                f"catalog_decoder={self.prob_intent_use_catalog_decoder}, "
                f"counterfactual_weight="
                f"{self.prob_intent_counterfactual_weight}, "
                f"scale={self.prob_intent_scale}, "
                f"inject_mode={self.prob_intent_inject_mode}, "
                f"shuffle={self.prob_intent_shuffle_condition}")

        self.use_collaborative_intent_memory = _as_bool(
            config.get('use_collaborative_intent_memory', False))
        self.collab_intent_scale = float(
            config.get('collab_intent_scale', 1.0))
        self.collab_intent_temperature = float(
            config.get('collab_intent_temperature', 0.1))
        self.collab_intent_recency_decay = float(
            config.get('collab_intent_recency_decay', 0.2))
        self.collab_intent_inject_mode = config.get(
            'collab_intent_inject_mode', 'target_mask')
        self.collab_intent_shuffle_condition = _as_bool(
            config.get('collab_intent_shuffle_condition', False))
        self.collab_intent_use_catalog_decoder = _as_bool(
            config.get('collab_intent_use_catalog_decoder', True))
        self.collab_intent_catalog_scale_multiplier = float(
            config.get(
                'collab_intent_catalog_scale_multiplier', 1.0))
        self.collab_intent_counterfactual_weight = float(
            config.get('collab_intent_counterfactual_weight', 0.0))
        self.collab_intent_counterfactual_margin = float(
            config.get('collab_intent_counterfactual_margin', 0.05))
        if self.use_collaborative_intent_memory:
            checkpoint_path = config.get(
                'collab_intent_checkpoint_path')
            if not checkpoint_path:
                raise ValueError(
                    'use_collaborative_intent_memory=True requires '
                    'collab_intent_checkpoint_path')
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False)
            item_embeddings = checkpoint[
                'backbone.item_embedding.weight'].float()
            codebook = checkpoint['codebook'].float()
            if item_embeddings.shape[0] != self.dataset.n_items:
                raise ValueError(
                    f'collaborative item embeddings '
                    f'{tuple(item_embeddings.shape)} do not match '
                    f'n_items={self.dataset.n_items}')
            item_embeddings = F.normalize(item_embeddings, dim=-1)
            codebook = F.normalize(codebook, dim=-1)
            item_code_scores = torch.matmul(
                item_embeddings, codebook.t())
            self.register_buffer(
                'collab_intent_item_embeddings',
                item_embeddings,
                persistent=False,
            )
            self.register_buffer(
                'collab_intent_codebook',
                codebook,
                persistent=False,
            )
            self.register_buffer(
                'collab_intent_item_code_scores',
                item_code_scores,
                persistent=False,
            )

            d_model = config['d_model']
            collaborative_dim = int(codebook.shape[-1])
            num_slots = int(config.get('collab_intent_num_slots', 4))
            num_heads = int(config.get(
                'collab_intent_num_heads', config['n_heads']))
            if num_slots > codebook.shape[0]:
                raise ValueError(
                    f'collab_intent_num_slots={num_slots} exceeds '
                    f'codebook size {codebook.shape[0]}')
            if d_model % num_heads != 0:
                raise ValueError(
                    f'd_model={d_model} must be divisible by '
                    f'collab_intent_num_heads={num_heads}')
            self.collab_intent_num_slots = num_slots
            self.collab_intent_projection = nn.Sequential(
                nn.LayerNorm(collaborative_dim),
                nn.Linear(collaborative_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.collab_intent_type_embedding = nn.Parameter(
                torch.zeros(num_slots, d_model))
            self.collab_intent_query_norm = nn.LayerNorm(d_model)
            self.collab_intent_cross_attention = nn.ModuleList([
                nn.MultiheadAttention(
                    d_model,
                    num_heads,
                    dropout=config['dropout_rate'],
                    batch_first=True,
                ) for _ in range(config['n_layers'])
            ])
            self.collab_intent_residual_norm = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(config['n_layers'])
            ])
            initial_gate = float(
                config.get(
                    'collab_intent_initial_residual_gate', 0.01))
            initial_gate = min(max(initial_gate, 1e-4), 1.0 - 1e-4)
            gate_logit = np.log(initial_gate / (1.0 - initial_gate))
            self.collab_intent_layer_gates = nn.Parameter(
                torch.full((config['n_layers'], ), float(gate_logit)))
            catalog_initial = float(
                config.get(
                    'collab_intent_catalog_initial_scale', 0.05))
            self.collab_intent_catalog_scales = nn.Parameter(
                torch.full(
                    (self.tokenizer.n_digit, ), catalog_initial))
            self.log(
                f"using collaborative multi-intent memory: "
                f"checkpoint={checkpoint_path}, "
                f"codes={codebook.shape[0]}, slots={num_slots}, "
                f"temperature={self.collab_intent_temperature}, "
                f"catalog_decoder="
                f"{self.collab_intent_use_catalog_decoder}, "
                f"catalog_multiplier="
                f"{self.collab_intent_catalog_scale_multiplier}, "
                f"counterfactual_weight="
                f"{self.collab_intent_counterfactual_weight}, "
                f"scale={self.collab_intent_scale}, "
                f"inject_mode={self.collab_intent_inject_mode}, "
                f"shuffle={self.collab_intent_shuffle_condition}")

        self.gen_steps = config['gen_steps']

        self.val_num_beams = config['val_num_beams']
        self.log(f"using num beams: {self.val_num_beams} for validation")
        self.num_beams = config['num_beams']

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(
                self.tokenizer.item2tokens[item])
        return item_id2tokens

    def get_posValidToken(self) -> torch.Tensor:
        posValidTokens = {}

        for item in self.tokenizer.item2tokens:
            cur_tokens = self.tokenizer.item2tokens[item]
            for pos in range(len(cur_tokens)):
                if pos not in posValidTokens.keys():
                    posValidTokens[pos] = set()
                posValidTokens[pos].add(cur_tokens[pos])
        max_token_num = 0

        for pos in posValidTokens:
            posValidTokens[pos] = sorted(list(posValidTokens[pos]))
            max_token_num = max(max_token_num, len(posValidTokens[pos]))

        posValidTokens_pt = torch.zeros((len(posValidTokens), max_token_num),
                                        dtype=torch.long)

        for pos in posValidTokens:
            posValidTokens_pt[
                pos, :len(posValidTokens[pos])] = torch.LongTensor(
                    posValidTokens[pos])

        return posValidTokens_pt

    def _build_sid_trie_children(self):
        children = {}
        for item in self.tokenizer.item2tokens:
            cur_tokens = tuple(int(x) for x in self.tokenizer.item2tokens[item])
            if not cur_tokens or all(x == 0 for x in cur_tokens):
                continue
            for depth in range(len(cur_tokens)):
                prefix = cur_tokens[:depth]
                children.setdefault(prefix, set()).add(cur_tokens[depth])
        return {prefix: sorted(tokens) for prefix, tokens in children.items()}

    def _build_semantic_token_bank(
        self,
        item_semantics: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Average catalog semantics for every position-specific SID token."""
        tokens = self.item_id2tokens.detach().cpu()
        semantics = item_semantics.detach().cpu()
        bank = semantics.new_zeros(
            (self.tokenizer.n_digit, vocab_size, semantics.shape[-1]))
        counts = semantics.new_zeros((self.tokenizer.n_digit, vocab_size, 1))
        valid_items = torch.arange(len(tokens)) > 0
        for position in range(self.tokenizer.n_digit):
            position_tokens = tokens[valid_items, position]
            position_semantics = semantics[valid_items]
            bank[position].index_add_(
                0, position_tokens, position_semantics)
            counts[position].index_add_(
                0,
                position_tokens,
                torch.ones(
                    (len(position_tokens), 1), dtype=semantics.dtype),
            )
        return bank / counts.clamp_min(1.0)

    def get_semantic_bridge_mask(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = input_ids != self.tokenizer.padding_token
        if self.semantic_bridge_inject_mode == 'target':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            valid_mask = valid_mask & inject_mask
        elif self.semantic_bridge_inject_mode == 'mask':
            valid_mask = valid_mask & (input_ids == self.mask_token_id)
        elif self.semantic_bridge_inject_mode == 'target_mask':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            valid_mask = (
                valid_mask
                & inject_mask
                & (input_ids == self.mask_token_id)
            )
        elif self.semantic_bridge_inject_mode != 'all':
            raise ValueError(
                f'Unknown semantic_bridge_inject_mode: '
                f'{self.semantic_bridge_inject_mode}')
        return valid_mask

    def get_semantic_bridge_context(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build an inference-safe semantic state from history SID tokens."""
        history_len = max(0, input_ids.shape[1] - self.tokenizer.n_digit)
        history = input_ids[:, :history_len]
        if history.shape[1] == 0:
            semantic_dim = self.semantic_bridge_token_bank.shape[-1]
            pooled = self.semantic_bridge_token_bank.new_zeros(
                (len(input_ids), semantic_dim))
            return self.semantic_bridge_context(pooled)

        positions = torch.arange(
            history.shape[1], device=history.device
        ) % self.tokenizer.n_digit
        safe_tokens = history.clamp(
            min=0, max=self.semantic_bridge_token_bank.shape[1] - 1)
        token_semantics = self.semantic_bridge_token_bank[
            positions.unsqueeze(0), safe_tokens]
        valid = (
            (history != self.tokenizer.padding_token)
            & (history != self.mask_token_id)
        )
        item_positions = (
            torch.arange(history.shape[1], device=history.device)
            // self.tokenizer.n_digit
        ).to(token_semantics.dtype)
        recency = torch.exp(0.15 * item_positions)[None]
        weights = valid.to(token_semantics.dtype) * recency
        pooled = (
            token_semantics * weights.unsqueeze(-1)
        ).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.semantic_bridge_shuffle_condition and len(pooled) > 1:
            pooled = pooled.roll(1, dims=0)
        return self.semantic_bridge_context(pooled)

    def build_semantic_bridge(
        self, input_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        context = self.get_semantic_bridge_context(input_ids)
        seq_context = context.unsqueeze(1).expand(
            -1, input_ids.shape[1], -1)
        residuals = [
            adapter(seq_context)
            for adapter in self.semantic_bridge_adapters
        ]
        return residuals, self.get_semantic_bridge_mask(input_ids)

    def semantic_bridge_alignment_loss(
        self,
        outputs: CausalLMOutputWithPast,
        masked_indices: torch.Tensor,
        label_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_hidden = outputs.hidden_states[-1][
            :, -self.tokenizer.n_digit:, :]
        weights = masked_indices.to(target_hidden.dtype)
        pooled = (
            target_hidden * weights.unsqueeze(-1)
        ).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        predicted = F.normalize(
            self.semantic_bridge_predictor(pooled), dim=-1)
        target = F.normalize(
            self.semantic_bridge_item_embeddings[
                label_item_ids.long().view(-1)],
            dim=-1,
        )
        cosine_loss = (1.0 - (predicted * target).sum(dim=-1)).mean()
        logits = torch.matmul(
            predicted, target.t()) / self.semantic_bridge_temperature
        contrastive_targets = torch.arange(
            len(predicted), device=predicted.device)
        contrastive_loss = F.cross_entropy(logits, contrastive_targets)
        return (
            cosine_loss
            + self.semantic_bridge_contrastive_weight * contrastive_loss
        )

    def get_multi_intent_mask(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = input_ids != self.tokenizer.padding_token
        if self.multi_intent_inject_mode == 'target':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            valid_mask = valid_mask & inject_mask
        elif self.multi_intent_inject_mode == 'mask':
            valid_mask = valid_mask & (input_ids == self.mask_token_id)
        elif self.multi_intent_inject_mode == 'target_mask':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            valid_mask = (
                valid_mask
                & inject_mask
                & (input_ids == self.mask_token_id)
            )
        elif self.multi_intent_inject_mode != 'all':
            raise ValueError(
                f'Unknown multi_intent_inject_mode: '
                f'{self.multi_intent_inject_mode}')
        return valid_mask

    def encode_multi_intent_memory(
        self, history_item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode exact history items into intent slots and mixture weights."""
        history_item_ids = history_item_ids.long().clamp(
            min=0, max=self.multi_intent_item_embeddings.shape[0] - 1)
        history_valid = history_item_ids != 0
        semantics = self.multi_intent_item_embeddings[history_item_ids]
        history = self.multi_intent_item_proj(semantics)

        history_len = history.shape[1]
        positions = torch.arange(
            history_len, device=history.device).clamp(
                max=self.multi_intent_history_pos.num_embeddings - 1)
        history = history + self.multi_intent_history_pos(
            positions).unsqueeze(0)

        all_padding = ~history_valid.any(dim=1)
        safe_padding_mask = ~history_valid
        if all_padding.any():
            safe_padding_mask = safe_padding_mask.clone()
            safe_padding_mask[all_padding, -1] = False

        queries = self.multi_intent_queries.unsqueeze(0).expand(
            history.shape[0], -1, -1)
        attended, _ = self.multi_intent_slot_attention(
            query=queries,
            key=history,
            value=history,
            key_padding_mask=safe_padding_mask,
            need_weights=False,
        )
        slots = self.multi_intent_slot_norm(queries + attended)
        probabilities = F.softmax(
            self.multi_intent_gate(slots).squeeze(-1), dim=-1)
        return slots, probabilities

    def build_multi_intent_control(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        history_item_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        slots, probabilities = self.encode_multi_intent_memory(
            history_item_ids)
        if self.multi_intent_shuffle_condition and len(slots) > 1:
            slots = slots.roll(1, dims=0)
            probabilities = probabilities.roll(1, dims=0)

        weighted_slots = (
            slots
            * probabilities.unsqueeze(-1)
            * self.multi_intent_num_slots
        )
        residuals = []
        for layer_idx, cross_attention in enumerate(
                self.multi_intent_cross_attention):
            attended, _ = cross_attention(
                query=inputs_embeds,
                key=weighted_slots,
                value=weighted_slots,
                need_weights=False,
            )
            gate = torch.sigmoid(
                self.multi_intent_layer_gates[layer_idx])
            residuals.append(
                gate
                * self.multi_intent_residual_norm[layer_idx](attended)
            )
        return (
            residuals,
            self.get_multi_intent_mask(input_ids),
            slots,
            probabilities,
        )

    def multi_intent_memory_alignment_loss(
        self,
        slots: torch.Tensor,
        probabilities: torch.Tensor,
        label_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next-item semantics using only history intent memory."""
        slot_semantics = F.normalize(
            self.multi_intent_semantic_decoder(slots), dim=-1)
        target = F.normalize(
            self.multi_intent_item_embeddings[
                label_item_ids.long().view(-1)],
            dim=-1,
        )

        mixture = F.normalize(
            (slot_semantics * probabilities.unsqueeze(-1)).sum(dim=1),
            dim=-1,
        )
        cosine_loss = (1.0 - (mixture * target).sum(dim=-1)).mean()

        slot_similarity = torch.einsum(
            'bkd,bd->bk', slot_semantics, target)
        assignment = F.softmax(
            torch.log(probabilities.clamp_min(1e-8))
            + slot_similarity / self.multi_intent_temperature,
            dim=-1,
        )
        assignment_loss = (
            1.0 - (assignment * slot_similarity).sum(dim=-1)
        ).mean()

        logits = torch.matmul(
            mixture, target.t()) / self.multi_intent_temperature
        contrastive_targets = torch.arange(
            len(mixture), device=mixture.device)
        contrastive_loss = F.cross_entropy(logits, contrastive_targets)

        normalized_slots = F.normalize(slots, dim=-1)
        pairwise = torch.matmul(
            normalized_slots, normalized_slots.transpose(1, 2))
        off_diagonal = ~torch.eye(
            self.multi_intent_num_slots,
            dtype=torch.bool,
            device=slots.device,
        ).unsqueeze(0)
        diversity_loss = pairwise[off_diagonal.expand_as(pairwise)].pow(
            2).mean()

        return (
            cosine_loss
            + self.multi_intent_assignment_weight * assignment_loss
            + self.multi_intent_contrastive_weight * contrastive_loss
            + self.multi_intent_diversity_weight * diversity_loss
        )

    def get_dialogue_multi_intent_mask(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = input_ids != self.tokenizer.padding_token
        if self.dialogue_multi_intent_inject_mode == 'target':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            return valid_mask & inject_mask
        if self.dialogue_multi_intent_inject_mode == 'mask':
            return valid_mask & (input_ids == self.mask_token_id)
        if self.dialogue_multi_intent_inject_mode == 'target_mask':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            return (
                valid_mask
                & inject_mask
                & (input_ids == self.mask_token_id)
            )
        if self.dialogue_multi_intent_inject_mode == 'all':
            return valid_mask
        raise ValueError(
            f'Unknown dialogue_multi_intent_inject_mode: '
            f'{self.dialogue_multi_intent_inject_mode}')

    def build_dialogue_multi_intent_control(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        user_ids, probabilities = (
            self.resolve_dialogue_multi_intent_condition(user_ids)
        )
        semantics = self.dialogue_multi_intent_embeddings[user_ids]
        slots = (
            self.dialogue_multi_intent_projection(semantics)
            + self.dialogue_multi_intent_slot_positions.unsqueeze(0)
        )
        weighted_slots = (
            slots
            * probabilities.unsqueeze(-1)
            * self.dialogue_multi_intent_num_slots
        )
        residuals = []
        for layer_idx, cross_attention in enumerate(
                self.dialogue_multi_intent_cross_attention):
            attended, _ = cross_attention(
                query=inputs_embeds,
                key=weighted_slots,
                value=weighted_slots,
                need_weights=False,
            )
            gate = torch.sigmoid(
                self.dialogue_multi_intent_layer_gates[layer_idx])
            residuals.append(
                self.dialogue_multi_intent_scale
                * gate
                * self.dialogue_multi_intent_residual_norm[layer_idx](
                    attended)
            )
        return (
            residuals,
            self.get_dialogue_multi_intent_mask(input_ids),
            slots,
            probabilities,
        )

    def resolve_dialogue_multi_intent_condition(
        self,
        user_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_ids = user_ids.long().clamp(
            min=0,
            max=self.dialogue_multi_intent_embeddings.shape[0] - 1,
        )
        if self.dialogue_multi_intent_shuffle_condition:
            num_users = self.dialogue_multi_intent_embeddings.shape[0] - 1
            offset = self.dialogue_multi_intent_shuffle_offset % num_users
            user_ids = torch.where(
                user_ids > 0,
                (user_ids - 1 + offset) % num_users + 1,
                user_ids,
            )
        probabilities = self.dialogue_multi_intent_probabilities[user_ids]
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        if self.dialogue_multi_intent_active_slot >= 0:
            active_slot = min(
                self.dialogue_multi_intent_active_slot,
                probabilities.shape[1] - 1,
            )
            slot_is_valid = probabilities[:, active_slot] > 0
            fallback_slots = probabilities.argmax(dim=-1)
            selected_slots = torch.where(
                slot_is_valid,
                torch.full_like(fallback_slots, active_slot),
                fallback_slots,
            )
            probabilities = F.one_hot(
                selected_slots, num_classes=probabilities.shape[1]
            ).to(probabilities.dtype)
        return user_ids, probabilities

    def dialogue_multi_intent_alignment_loss(
        self,
        slots: torch.Tensor,
        probabilities: torch.Tensor,
        label_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        slot_semantics = F.normalize(
            self.dialogue_multi_intent_decoder(slots), dim=-1)
        target = F.normalize(
            self.dialogue_multi_intent_item_embeddings[
                label_item_ids.long().view(-1)],
            dim=-1,
        )
        similarities = torch.einsum(
            'bkd,bd->bk', slot_semantics, target)
        assignment = F.softmax(
            torch.log(probabilities.clamp_min(1e-8))
            + similarities / self.dialogue_multi_intent_temperature,
            dim=-1,
        )
        assignment_loss = (
            1.0 - (assignment * similarities).sum(dim=-1)
        ).mean()
        mixture = F.normalize(
            (slot_semantics * assignment.unsqueeze(-1)).sum(dim=1),
            dim=-1,
        )
        cosine_loss = (1.0 - (mixture * target).sum(dim=-1)).mean()
        logits = torch.matmul(
            mixture, target.t()) / self.dialogue_multi_intent_temperature
        contrastive_targets = torch.arange(
            len(mixture), device=mixture.device)
        contrastive_loss = F.cross_entropy(logits, contrastive_targets)
        pairwise = torch.matmul(
            slot_semantics, slot_semantics.transpose(1, 2))
        off_diagonal = ~torch.eye(
            self.dialogue_multi_intent_num_slots,
            dtype=torch.bool,
            device=slots.device,
        ).unsqueeze(0)
        diversity_loss = pairwise[
            off_diagonal.expand_as(pairwise)].pow(2).mean()
        return (
            cosine_loss
            + self.dialogue_multi_intent_assignment_weight * assignment_loss
            + self.dialogue_multi_intent_contrastive_weight * contrastive_loss
            + self.dialogue_multi_intent_diversity_weight * diversity_loss
        )

    def get_prob_intent_mask(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = input_ids != self.tokenizer.padding_token
        if self.prob_intent_inject_mode == 'target':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            return valid_mask & inject_mask
        if self.prob_intent_inject_mode == 'mask':
            return valid_mask & (input_ids == self.mask_token_id)
        if self.prob_intent_inject_mode == 'target_mask':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            return (
                valid_mask
                & inject_mask
                & (input_ids == self.mask_token_id)
            )
        if self.prob_intent_inject_mode != 'all':
            raise ValueError(
                f'Unknown prob_intent_inject_mode: '
                f'{self.prob_intent_inject_mode}')
        return valid_mask

    def build_history_category_distribution(
        self, history_item_ids: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[list[torch.Tensor]],
    ]:
        history_item_ids = history_item_ids.long().clamp(
            min=0, max=self.prob_intent_item_categories.shape[0] - 1)
        if self.prob_intent_learned_router is not None:
            self.prob_intent_learned_router.eval()
            with torch.no_grad():
                logits, hierarchy_logits = self.prob_intent_learned_router(
                    history_item_ids)
                distribution = F.softmax(logits, dim=-1)
                hierarchy_distributions = [
                    F.softmax(level_logits, dim=-1)
                    for level_logits in hierarchy_logits
                ]
            probabilities, category_ids = distribution.topk(
                self.prob_intent_num_slots, dim=-1)
            slot_mask = probabilities > 0
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True).clamp_min(1e-8)
            return (
                category_ids,
                probabilities,
                slot_mask,
                distribution,
                hierarchy_distributions,
            )
        valid = history_item_ids != 0
        categories = self.prob_intent_item_categories[history_item_ids]
        positions = torch.arange(
            history_item_ids.shape[1],
            device=history_item_ids.device,
            dtype=torch.float32,
        )
        recency = torch.exp(self.prob_intent_recency_decay * positions)
        weights = recency.unsqueeze(0) * valid
        distribution = torch.zeros(
            history_item_ids.shape[0],
            self.prob_intent_category_tokens.shape[0],
            device=history_item_ids.device,
            dtype=torch.float32,
        )
        distribution.scatter_add_(1, categories, weights)
        distribution = distribution / distribution.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        if self.prob_intent_category_transition is not None:
            transition = self.prob_intent_category_transition
            if self.prob_intent_trainable_router:
                transition = F.softmax(
                    self.prob_intent_transition_logits, dim=-1)
            distribution = torch.matmul(
                distribution,
                transition.to(distribution.dtype),
            )
        probabilities, category_ids = distribution.topk(
            self.prob_intent_num_slots, dim=-1)
        slot_mask = probabilities > 0
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        return category_ids, probabilities, slot_mask, distribution, None

    def build_probabilistic_intent_control(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        history_item_ids: torch.Tensor,
        shuffle_override: Optional[bool] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        (category_ids, probabilities, slot_mask, distribution,
         hierarchy_distributions) = (
             self.build_history_category_distribution(history_item_ids))
        should_shuffle = (
            self.prob_intent_shuffle_condition
            if shuffle_override is None
            else shuffle_override
        )
        if should_shuffle:
            category_ids = (
                category_ids + 137
            ) % self.prob_intent_category_tokens.shape[0]
            distribution = torch.roll(
                distribution, shifts=137, dims=-1)
            if hierarchy_distributions is not None:
                hierarchy_distributions = [
                    torch.roll(level_distribution, shifts=shift, dims=-1)
                    for level_distribution, shift in zip(
                        hierarchy_distributions, (7, 31, 67))
                ]

        hierarchy_sid_bias = None
        if (self.prob_intent_use_hierarchy_sid_prior
                and hierarchy_distributions is not None):
            hierarchy_biases = []
            for hierarchy_depth, level_distribution in enumerate(
                    hierarchy_distributions):
                hierarchy_llr = getattr(
                    self,
                    f'prob_intent_hierarchy_sid_llr_{hierarchy_depth}',
                )
                hierarchy_biases.append(torch.einsum(
                    'bc,cdv->bdv',
                    level_distribution.to(hierarchy_llr.dtype),
                    hierarchy_llr,
                ))
            hierarchy_sid_bias = torch.stack(
                hierarchy_biases, dim=0).mean(dim=0)

        category_tokens = self.prob_intent_category_tokens[
            category_ids].mean(dim=2)
        dimension = category_tokens.shape[-1]
        frequencies = torch.exp(
            -math.log(1000.0)
            * torch.arange(
                0,
                dimension,
                2,
                device=category_tokens.device,
                dtype=category_tokens.dtype,
            )
            / max(dimension, 2)
        )
        angles = probabilities.unsqueeze(-1) * frequencies * (
            2 * math.pi)
        probability_encoding = torch.zeros_like(category_tokens)
        probability_encoding[..., 0::2] = angles.sin()
        probability_encoding[..., 1::2] = angles.cos()[
            ..., :probability_encoding[..., 1::2].shape[-1]]
        category_tokens = (
            category_tokens + 0.25 * probability_encoding
        ) * slot_mask.unsqueeze(-1)
        memory = (
            self.prob_intent_text_projection(
                self.prob_intent_text_norm(category_tokens))
            + self.prob_intent_type_embedding.unsqueeze(0)
        )
        memory = (
            memory
            * probabilities.unsqueeze(-1)
            * self.prob_intent_num_slots
        )

        query = self.prob_intent_query_norm(inputs_embeds)
        residuals = []
        for layer_idx, cross_attention in enumerate(
                self.prob_intent_cross_attention):
            attended, _ = cross_attention(
                query=query,
                key=memory,
                value=memory,
                key_padding_mask=~slot_mask,
                need_weights=False,
            )
            gate = torch.sigmoid(
                self.prob_intent_layer_gates[layer_idx])
            residuals.append(
                gate
                * self.prob_intent_residual_norm[layer_idx](attended)
            )
        return (
            residuals,
            self.get_prob_intent_mask(input_ids),
            category_ids,
            probabilities,
            distribution,
            hierarchy_sid_bias,
        )

    def build_catalog_intent_sid_bias(
        self,
        target_tokens: torch.Tensor,
        category_ids: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Decode category intent through catalog items into SID-token bias."""
        catalog_tokens = self.item_id2tokens[1:].to(target_tokens.device)
        catalog_categories = self.prob_intent_item_categories[1:]
        item_probabilities = (
            catalog_categories[None, :, None]
            == category_ids[:, None, :]
        ).to(probabilities.dtype)
        item_probabilities = (
            item_probabilities * probabilities[:, None, :]
        ).sum(dim=-1)

        revealed = (
            (target_tokens != self.mask_token_id)
            & (target_tokens != self.tokenizer.padding_token)
        )
        catalog_match = (
            (catalog_tokens[None, :, :] == target_tokens[:, None, :])
            | ~revealed[:, None, :]
        ).all(dim=-1)
        valid_prefix = catalog_match.any(dim=-1, keepdim=True)
        candidate_weights = item_probabilities * catalog_match
        global_mean = item_probabilities.mean(
            dim=-1, keepdim=True).clamp_min(1e-6)

        batch_size = target_tokens.shape[0]
        vocab_size = self.tokenizer.vocab_size
        bias = probabilities.new_zeros(
            (batch_size, self.tokenizer.n_digit, vocab_size))
        candidate_float = catalog_match.to(probabilities.dtype)
        for depth in range(self.tokenizer.n_digit):
            token_ids = catalog_tokens[:, depth].unsqueeze(0).expand(
                batch_size, -1)
            token_count = probabilities.new_zeros(
                (batch_size, vocab_size))
            token_mass = probabilities.new_zeros(
                (batch_size, vocab_size))
            token_count.scatter_add_(
                dim=1, index=token_ids, src=candidate_float)
            token_mass.scatter_add_(
                dim=1, index=token_ids, src=candidate_weights)
            token_average = token_mass / token_count.clamp_min(1.0)
            depth_bias = torch.log(
                token_average.clamp_min(1e-6) / global_mean)
            depth_bias = depth_bias.clamp(min=-3.0, max=3.0)
            depth_bias = depth_bias.masked_fill(token_count == 0, -3.0)
            depth_bias = depth_bias * valid_prefix
            bias[:, depth, :] = depth_bias
        return bias

    def get_collab_intent_mask(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = input_ids != self.tokenizer.padding_token
        if self.collab_intent_inject_mode == 'target':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            return valid_mask & inject_mask
        if self.collab_intent_inject_mode == 'mask':
            return valid_mask & (input_ids == self.mask_token_id)
        if self.collab_intent_inject_mode == 'target_mask':
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -self.tokenizer.n_digit:] = True
            return (
                valid_mask
                & inject_mask
                & (input_ids == self.mask_token_id)
            )
        if self.collab_intent_inject_mode != 'all':
            raise ValueError(
                f'Unknown collab_intent_inject_mode: '
                f'{self.collab_intent_inject_mode}')
        return valid_mask

    def build_collaborative_intent_control(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        history_item_ids: torch.Tensor,
        shuffle_override: Optional[bool] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor,
               torch.Tensor, torch.Tensor]:
        history_item_ids = history_item_ids.long().clamp(
            min=0, max=self.collab_intent_item_embeddings.shape[0] - 1)
        valid = history_item_ids != 0
        assignments = F.softmax(
            self.collab_intent_item_code_scores[history_item_ids]
            / self.collab_intent_temperature,
            dim=-1,
        )
        positions = torch.arange(
            history_item_ids.shape[1],
            device=history_item_ids.device,
            dtype=assignments.dtype,
        )
        recency = torch.exp(
            self.collab_intent_recency_decay * positions)
        weights = recency.unsqueeze(0) * valid
        weights = weights / weights.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        distribution = (
            assignments * weights.unsqueeze(-1)
        ).sum(dim=1)
        probabilities, code_ids = distribution.topk(
            self.collab_intent_num_slots, dim=-1)
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)

        should_shuffle = (
            self.collab_intent_shuffle_condition
            if shuffle_override is None
            else shuffle_override
        )
        if should_shuffle:
            code_ids = (
                code_ids + 137
            ) % self.collab_intent_codebook.shape[0]

        memory = (
            self.collab_intent_projection(
                self.collab_intent_codebook[code_ids])
            + self.collab_intent_type_embedding.unsqueeze(0)
        )
        memory = (
            memory
            * probabilities.unsqueeze(-1)
            * self.collab_intent_num_slots
        )
        query = self.collab_intent_query_norm(inputs_embeds)
        residuals = []
        for layer_idx, cross_attention in enumerate(
                self.collab_intent_cross_attention):
            attended, _ = cross_attention(
                query=query,
                key=memory,
                value=memory,
                need_weights=False,
            )
            gate = torch.sigmoid(
                self.collab_intent_layer_gates[layer_idx])
            residuals.append(
                gate
                * self.collab_intent_residual_norm[layer_idx](attended)
            )
        return (
            residuals,
            self.get_collab_intent_mask(input_ids),
            code_ids,
            probabilities,
        )

    def build_collaborative_catalog_sid_bias(
        self,
        target_tokens: torch.Tensor,
        code_ids: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Ground collaborative intent scores through valid catalog items."""
        catalog_tokens = self.item_id2tokens[1:].to(target_tokens.device)
        selected_scores = self.collab_intent_item_code_scores[
            1:][:, code_ids].permute(1, 0, 2)
        item_scores = (
            selected_scores * probabilities[:, None, :]
        ).sum(dim=-1)

        revealed = (
            (target_tokens != self.mask_token_id)
            & (target_tokens != self.tokenizer.padding_token)
        )
        catalog_match = (
            (catalog_tokens[None, :, :] == target_tokens[:, None, :])
            | ~revealed[:, None, :]
        ).all(dim=-1)
        valid_prefix = catalog_match.any(dim=-1, keepdim=True)
        candidate_scores = item_scores * catalog_match
        global_mean = item_scores.mean(dim=-1, keepdim=True)
        global_std = item_scores.std(
            dim=-1, keepdim=True).clamp_min(1e-4)

        batch_size = target_tokens.shape[0]
        vocab_size = self.tokenizer.vocab_size
        bias = probabilities.new_zeros(
            (batch_size, self.tokenizer.n_digit, vocab_size))
        candidate_float = catalog_match.to(probabilities.dtype)
        for depth in range(self.tokenizer.n_digit):
            token_ids = catalog_tokens[:, depth].unsqueeze(0).expand(
                batch_size, -1)
            token_count = probabilities.new_zeros(
                (batch_size, vocab_size))
            token_score = probabilities.new_zeros(
                (batch_size, vocab_size))
            token_count.scatter_add_(
                dim=1, index=token_ids, src=candidate_float)
            token_score.scatter_add_(
                dim=1, index=token_ids, src=candidate_scores)
            token_average = token_score / token_count.clamp_min(1.0)
            depth_bias = (
                token_average - global_mean
            ) / global_std
            depth_bias = depth_bias.clamp(min=-3.0, max=3.0)
            depth_bias = depth_bias.masked_fill(token_count == 0, -3.0)
            bias[:, depth, :] = depth_bias * valid_prefix
        return bias

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        emb_params = sum(
            p.numel() for p in self.llada.get_input_embeddings().parameters()
            if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def add_item_pos_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        item_seq_len = input_ids.shape[1] // self.tokenizer.n_digit
        item_pos_idx = torch.arange(item_seq_len, device=input_ids.device)

        token_pos_idx = item_pos_idx.repeat_interleave(
            repeats=self.tokenizer.n_digit, dim=0)

        token_pos_emb = self.item_pos_emb(token_pos_idx)
        token_pos_emb = token_pos_emb.unsqueeze(0).expand(
            (input_ids.shape[0], -1, -1))

        input_embeds = self.llada.model.transformer.wte(input_ids)

        input_embeds = input_embeds + token_pos_emb

        return input_embeds

    def get_user_code_context(self,
                              input_ids: torch.Tensor,
                              input_embeds: torch.Tensor,
                              return_aux: bool = False):
        history_len = self.config['max_item_seq_len'] * self.tokenizer.n_digit
        pool_len = min(history_len, input_ids.shape[1])
        pool_ids = input_ids[:, :pool_len]
        pool_embeds = input_embeds[:, :pool_len, :]
        pool_mask = (pool_ids != self.tokenizer.padding_token).unsqueeze(-1)

        denom = pool_mask.sum(dim=1).clamp_min(1)
        user_hidden = (pool_embeds * pool_mask).sum(dim=1) / denom

        user_query = F.normalize(user_hidden, dim=-1)
        codebook = F.normalize(self.user_codebook, dim=-1)
        assign_logits = torch.matmul(user_query, codebook.t())
        intent_context = None
        if getattr(self, 'use_intent_codebook', False):
            intent_bank = self.intent_proj(self.intent_codebook)
            intent_query = F.normalize(self.intent_query(user_hidden), dim=-1)
            intent_bank_norm = F.normalize(intent_bank, dim=-1)
            intent_logits = torch.matmul(intent_query, intent_bank_norm.t())
            assign_logits = assign_logits + self.intent_codebook_weight * \
                intent_logits
        assign = F.softmax(assign_logits / self.user_code_temperature, dim=-1)
        code_context = torch.matmul(assign, self.user_codebook)
        if getattr(self, 'use_intent_codebook', False):
            intent_context = torch.matmul(assign, intent_bank)

        gate_inputs = [user_hidden, code_context]
        if intent_context is not None:
            gate_inputs.append(intent_context)
        code_context = self.user_code_gate(torch.cat(gate_inputs, dim=-1))
        code_context = self.user_code_norm(code_context)

        if return_aux:
            return code_context, assign_logits, assign
        return code_context

    def add_user_code_context(self, input_ids: torch.Tensor,
                              input_embeds: torch.Tensor,
                              code_context: torch.Tensor) -> torch.Tensor:
        valid_mask = input_ids != self.tokenizer.padding_token
        if self.user_code_inject_mode == 'target':
            target_len = self.tokenizer.n_digit
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -target_len:] = True
            valid_mask = valid_mask & inject_mask
        elif self.user_code_inject_mode == 'mask':
            valid_mask = valid_mask & (input_ids == self.mask_token_id)
        elif self.user_code_inject_mode == 'target_mask':
            target_len = self.tokenizer.n_digit
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -target_len:] = True
            valid_mask = valid_mask & inject_mask & (input_ids
                                                     == self.mask_token_id)
        elif self.user_code_inject_mode != 'all':
            raise ValueError(
                f'Unknown user_code_inject_mode: {self.user_code_inject_mode}')

        return input_embeds + self.user_code_scale * code_context.unsqueeze(
            1) * valid_mask.unsqueeze(-1)

    def get_qwen_reasoning_context(
        self, user_ids: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if user_ids is None:
            return None
        user_ids = user_ids.long().clamp(
            min=0, max=self.qwen_reasoning_user_code_ids.shape[0] - 1)
        code_ids = self.qwen_reasoning_user_code_ids[user_ids]
        valid = code_ids >= 0
        code_ids = code_ids.clamp(
            min=0, max=self.qwen_reasoning_code_emb.shape[0] - 1)
        reason_emb = self.qwen_reasoning_code_emb[code_ids]
        reason_context = self.qwen_reasoning_adapter(reason_emb)
        reason_context = self.qwen_reasoning_norm(reason_context)
        return reason_context * valid.unsqueeze(-1)

    def add_qwen_reasoning_context(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        reason_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if reason_context is None:
            return input_embeds

        valid_mask = input_ids != self.tokenizer.padding_token
        if self.qwen_reasoning_inject_mode == 'target':
            target_len = self.tokenizer.n_digit
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -target_len:] = True
            valid_mask = valid_mask & inject_mask
        elif self.qwen_reasoning_inject_mode == 'mask':
            valid_mask = valid_mask & (input_ids == self.mask_token_id)
        elif self.qwen_reasoning_inject_mode == 'target_mask':
            target_len = self.tokenizer.n_digit
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -target_len:] = True
            valid_mask = valid_mask & inject_mask & (input_ids
                                                     == self.mask_token_id)
        elif self.qwen_reasoning_inject_mode != 'all':
            raise ValueError(
                f'Unknown qwen_reasoning_inject_mode: {self.qwen_reasoning_inject_mode}'
            )

        return input_embeds + self.qwen_reasoning_scale * \
            reason_context.unsqueeze(1) * valid_mask.unsqueeze(-1)

    def get_rec_control_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        valid_mask = input_ids != self.tokenizer.padding_token
        if self.rec_control_inject_mode == 'target':
            target_len = self.tokenizer.n_digit
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -target_len:] = True
            valid_mask = valid_mask & inject_mask
        elif self.rec_control_inject_mode == 'mask':
            valid_mask = valid_mask & (input_ids == self.mask_token_id)
        elif self.rec_control_inject_mode == 'target_mask':
            target_len = self.tokenizer.n_digit
            inject_mask = torch.zeros_like(valid_mask)
            inject_mask[:, -target_len:] = True
            valid_mask = valid_mask & inject_mask & (input_ids
                                                     == self.mask_token_id)
        elif self.rec_control_inject_mode != 'all':
            raise ValueError(
                f'Unknown rec_control_inject_mode: {self.rec_control_inject_mode}'
            )
        return valid_mask

    def build_rec_control(
        self,
        input_ids: torch.Tensor,
        code_context: Optional[torch.Tensor],
        reason_context: Optional[torch.Tensor],
    ):
        condition_context = self.get_rec_control_condition(code_context,
                                                           reason_context)
        if condition_context is None:
            return None, None

        seq_condition = condition_context.unsqueeze(1).expand(
            -1, input_ids.shape[1], -1)
        control_residuals = [
            adapter(seq_condition)
            for adapter in self.rec_control_adapters
        ]
        control_mask = self.get_rec_control_mask(input_ids)
        return control_residuals, control_mask

    def get_rec_control_condition(
        self,
        code_context: Optional[torch.Tensor],
        reason_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        condition_parts = []
        if self.rec_control_use_user_code and code_context is not None:
            condition_parts.append(code_context)
        if self.rec_control_use_qwen and reason_context is not None:
            condition_parts.append(reason_context)
        if not condition_parts:
            return None
        condition_context = torch.stack(condition_parts, dim=0).sum(dim=0)
        return self.rec_control_norm(condition_context)

    def llada_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        history_item_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        disable_user_conditions: bool = False,
        prob_intent_shuffle_override: Optional[bool] = None,
        collab_intent_shuffle_override: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = self.add_item_pos_emb(input_ids)
        code_context = None
        reason_context = None
        control_residuals = None
        control_mask = None
        condition_keep = None
        multi_intent_slots = None
        multi_intent_probabilities = None
        dialogue_multi_intent_slots = None
        dialogue_multi_intent_probabilities = None
        prob_intent_category_ids = None
        prob_intent_probabilities = None
        prob_intent_distribution = None
        prob_intent_hierarchy_sid_bias = None
        prob_intent_mask = None
        prob_intent_target_tokens = None
        collab_intent_code_ids = None
        collab_intent_probabilities = None
        collab_intent_mask = None
        collab_intent_target_tokens = None
        if (self.training and not disable_user_conditions
                and self.user_condition_dropout > 0.0):
            keep_prob = 1.0 - self.user_condition_dropout
            condition_keep = (
                torch.rand(input_ids.shape[0],
                           device=input_ids.device) < keep_prob).to(
                               inputs_embeds.dtype).unsqueeze(-1)
        apply_input_conditions = not (
            self.use_rec_controlnet
            and self.rec_control_disable_input_injection)
        if self.use_user_codebook and not disable_user_conditions:
            code_context = self.get_user_code_context(input_ids, inputs_embeds)
            if condition_keep is not None:
                code_context = code_context * condition_keep
            if apply_input_conditions:
                inputs_embeds = self.add_user_code_context(input_ids,
                                                           inputs_embeds,
                                                           code_context)
        if self.use_qwen_reasoning_adapter and not disable_user_conditions:
            reason_context = self.get_qwen_reasoning_context(user_ids)
            if reason_context is not None and condition_keep is not None:
                reason_context = reason_context * condition_keep
            if apply_input_conditions:
                inputs_embeds = self.add_qwen_reasoning_context(
                    input_ids, inputs_embeds, reason_context)
        if self.use_rec_controlnet and not disable_user_conditions:
            control_residuals, control_mask = self.build_rec_control(
                input_ids, code_context, reason_context)
        if self.use_semantic_bridge and not disable_user_conditions:
            semantic_residuals, semantic_mask = \
                self.build_semantic_bridge(input_ids)
            if control_residuals is None:
                control_residuals = semantic_residuals
                control_mask = semantic_mask
            else:
                control_residuals = [
                    control * control_mask.unsqueeze(-1)
                    + semantic * semantic_mask.unsqueeze(-1)
                    for control, semantic in zip(
                        control_residuals, semantic_residuals)
                ]
                control_mask = attention_mask.bool()
        if (self.use_dialogue_multi_intent
                and not disable_user_conditions
                and user_ids is not None):
            (dialogue_residuals, dialogue_intent_mask,
             dialogue_multi_intent_slots,
             dialogue_multi_intent_probabilities) = (
                 self.build_dialogue_multi_intent_control(
                     input_ids=input_ids,
                     inputs_embeds=inputs_embeds,
                     user_ids=user_ids,
                 )
             )
            if condition_keep is not None:
                dialogue_residuals = [
                    residual * condition_keep.unsqueeze(-1)
                    for residual in dialogue_residuals
                ]
            if control_residuals is None:
                control_residuals = dialogue_residuals
                control_mask = dialogue_intent_mask
            else:
                control_residuals = [
                    control * control_mask.unsqueeze(-1)
                    + residual * dialogue_intent_mask.unsqueeze(-1)
                    for control, residual in zip(
                        control_residuals, dialogue_residuals)
                ]
                control_mask = attention_mask.bool()
        if (self.use_multi_intent_memory
                and not disable_user_conditions
                and history_item_ids is not None):
            (intent_residuals, intent_mask, multi_intent_slots,
             multi_intent_probabilities) = self.build_multi_intent_control(
                 input_ids=input_ids,
                 inputs_embeds=inputs_embeds,
                 history_item_ids=history_item_ids,
             )
            if control_residuals is None:
                control_residuals = intent_residuals
                control_mask = intent_mask
            else:
                control_residuals = [
                    control * control_mask.unsqueeze(-1)
                    + intent * intent_mask.unsqueeze(-1)
                    for control, intent in zip(
                        control_residuals, intent_residuals)
                ]
                control_mask = attention_mask.bool()
        if (self.use_probabilistic_intent_memory
                and not disable_user_conditions
                and history_item_ids is not None):
            (prob_residuals, prob_intent_mask, prob_intent_category_ids,
             prob_intent_probabilities, prob_intent_distribution,
             prob_intent_hierarchy_sid_bias) = (
                 self.build_probabilistic_intent_control(
                     input_ids=input_ids,
                     inputs_embeds=inputs_embeds,
                     history_item_ids=history_item_ids,
                     shuffle_override=prob_intent_shuffle_override,
                )
             )
            prob_intent_target_tokens = input_ids[
                :, -self.tokenizer.n_digit:].clone()
            if control_residuals is None:
                control_residuals = prob_residuals
                control_mask = prob_intent_mask
            else:
                control_residuals = [
                    control * control_mask.unsqueeze(-1)
                    + residual * prob_intent_mask.unsqueeze(-1)
                    for control, residual in zip(
                        control_residuals, prob_residuals)
                ]
                control_mask = attention_mask.bool()
        if (self.use_collaborative_intent_memory
                and not disable_user_conditions
                and history_item_ids is not None):
            (collab_residuals, collab_intent_mask,
             collab_intent_code_ids,
             collab_intent_probabilities) = (
                 self.build_collaborative_intent_control(
                     input_ids=input_ids,
                     inputs_embeds=inputs_embeds,
                     history_item_ids=history_item_ids,
                     shuffle_override=collab_intent_shuffle_override,
                 )
             )
            collab_intent_target_tokens = input_ids[
                :, -self.tokenizer.n_digit:].clone()
            if control_residuals is None:
                control_residuals = collab_residuals
                control_mask = collab_intent_mask
            else:
                control_residuals = [
                    control * control_mask.unsqueeze(-1)
                    + residual * collab_intent_mask.unsqueeze(-1)
                    for control, residual in zip(
                        control_residuals, collab_residuals)
                ]
                control_mask = attention_mask.bool()
        input_ids = None

        outputs = self.llada(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            output_hidden_states=True,
            control_residuals=control_residuals,
            control_mask=control_mask,
            control_scale=(
                self.collab_intent_scale
                if self.use_collaborative_intent_memory
                and not self.use_probabilistic_intent_memory
                and not self.use_multi_intent_memory
                and not self.use_semantic_bridge
                and not self.use_rec_controlnet
                else self.prob_intent_scale
                if self.use_probabilistic_intent_memory
                and not self.use_multi_intent_memory
                and not self.use_semantic_bridge
                and not self.use_rec_controlnet
                else self.multi_intent_scale
                if self.use_multi_intent_memory
                and not self.use_semantic_bridge
                and not self.use_rec_controlnet
                else (
                    self.semantic_bridge_scale
                    if self.use_semantic_bridge
                    and not self.use_rec_controlnet
                    else self.rec_control_scale
                )
            ),
        )
        outputs.multi_intent_slots = multi_intent_slots
        outputs.multi_intent_probabilities = multi_intent_probabilities
        outputs.dialogue_multi_intent_slots = dialogue_multi_intent_slots
        outputs.dialogue_multi_intent_probabilities = (
            dialogue_multi_intent_probabilities)
        outputs.prob_intent_category_ids = prob_intent_category_ids
        outputs.prob_intent_probabilities = prob_intent_probabilities
        outputs.prob_intent_distribution = prob_intent_distribution
        outputs.collab_intent_code_ids = collab_intent_code_ids
        outputs.collab_intent_probabilities = collab_intent_probabilities
        if (self.use_probabilistic_intent_memory
                and self.prob_intent_use_sid_prior
                and prob_intent_category_ids is not None):
            if (self.prob_intent_use_hierarchy_sid_prior
                    and prob_intent_hierarchy_sid_bias is not None):
                sid_bias = prob_intent_hierarchy_sid_bias
            elif self.prob_intent_use_full_distribution_sid_prior:
                sid_bias = torch.einsum(
                    'bc,cdv->bdv',
                    prob_intent_distribution.to(
                        self.prob_intent_category_sid_llr.dtype),
                    self.prob_intent_category_sid_llr,
                )
            else:
                category_sid_llr = self.prob_intent_category_sid_llr[
                    prob_intent_category_ids]
                sid_bias = (
                    category_sid_llr
                    * prob_intent_probabilities[:, :, None, None]
                ).sum(dim=1)
            target_mask = prob_intent_mask[:, -self.tokenizer.n_digit:]
            scaled_bias = (
                sid_bias
                * self.prob_intent_sid_prior_scales[None, :, None]
                * target_mask.unsqueeze(-1)
            )
            outputs.logits = outputs.logits.clone()
            outputs.logits[:, -self.tokenizer.n_digit:, :] += scaled_bias
        if (self.use_probabilistic_intent_memory
                and self.prob_intent_use_catalog_decoder
                and prob_intent_category_ids is not None):
            catalog_bias = self.build_catalog_intent_sid_bias(
                target_tokens=prob_intent_target_tokens,
                category_ids=prob_intent_category_ids,
                probabilities=prob_intent_probabilities,
            )
            target_mask = prob_intent_mask[:, -self.tokenizer.n_digit:]
            scaled_catalog_bias = (
                catalog_bias
                * self.prob_intent_catalog_scales[None, :, None]
                * target_mask.unsqueeze(-1)
            )
            outputs.logits = outputs.logits.clone()
            outputs.logits[:, -self.tokenizer.n_digit:, :] += (
                scaled_catalog_bias
            )
        if (self.use_collaborative_intent_memory
                and self.collab_intent_use_catalog_decoder
                and collab_intent_code_ids is not None):
            catalog_bias = self.build_collaborative_catalog_sid_bias(
                target_tokens=collab_intent_target_tokens,
                code_ids=collab_intent_code_ids,
                probabilities=collab_intent_probabilities,
            )
            target_mask = collab_intent_mask[
                :, -self.tokenizer.n_digit:]
            scaled_catalog_bias = (
                catalog_bias
                * self.collab_intent_catalog_scales[None, :, None]
                * self.collab_intent_catalog_scale_multiplier
                * target_mask.unsqueeze(-1)
            )
            outputs.logits = outputs.logits.clone()
            outputs.logits[:, -self.tokenizer.n_digit:, :] += (
                scaled_catalog_bias
            )
        if (self.use_user_codebook and self.use_user_code_logits_bias
                and code_context is not None):
            logits_bias = self.user_code_logits_bias(code_context)
            outputs.logits = outputs.logits + self.user_code_logits_scale * \
                logits_bias.unsqueeze(1)

        return outputs

    def his_mask_loss(
        self,
        input_tokens,
        input_attention_mask,
    ) -> CausalLMOutputWithPast:
        masked_input_tokens, masked_indices, p_mask = self.get_masked_seqs(
            input_tokens, input_attention_mask)

        outputs = self.llada_forward(
            input_ids=masked_input_tokens,
            attention_mask=input_attention_mask.long())

        logits = outputs.logits
        logits = logits / self.temperature

        input_tokens = input_tokens.masked_fill(~input_attention_mask,
                                                self.tokenizer.ignored_label)

        token_loss = self.loss_fct(
            logits[masked_indices],
            input_tokens[masked_indices]) / p_mask[masked_indices]
        loss = token_loss.sum() / (input_tokens.shape[0] *
                                   input_tokens.shape[1])
        outputs.loss = loss

        return outputs

    def target_mask_loss(
        self,
        input_tokens,
        input_attention_mask,
        label_tokens,
        label_attention_mask,
        user_ids=None,
        label_item_ids=None,
        history_item_ids=None,
    ) -> CausalLMOutputWithPast:

        masked_label_tokens, masked_indices, p_mask = self.get_masked_seqs(
            label_tokens, label_attention_mask, is_target=True)

        final_inputs_ids = torch.cat([input_tokens, masked_label_tokens],
                                     dim=1)
        final_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)

        outputs = self.llada_forward(
            input_ids=final_inputs_ids,
            attention_mask=final_attention_mask.long(),
            user_ids=user_ids,
            history_item_ids=history_item_ids)

        clean_label_tokens = label_tokens
        sid_anchor_logits = None
        sid_anchor_teacher_logits = None
        sid_anchor_wrong_target_logits = None
        if (self.use_sid_anchor_condition
                and history_item_ids is not None):
            sid_anchor_logits = self.get_sid_anchor_logits(
                history_item_ids)
            if self.sid_anchor_teacher_router is not None:
                self.sid_anchor_teacher_router.eval()
                with torch.no_grad():
                    sid_anchor_teacher_logits = (
                        self.sid_anchor_teacher_router(
                            history_item_ids.long()))
            keep_mask = None
            if self.training and self.sid_anchor_dropout > 0.0:
                keep_mask = (
                    torch.rand(
                        history_item_ids.shape[0],
                        self.tokenizer.n_digit,
                        device=history_item_ids.device,
                    ) >= self.sid_anchor_dropout
                ).to(outputs.logits.dtype)
                selected = keep_mask[:, self.sid_anchor_digits]
                empty = selected.sum(dim=-1) == 0
                if empty.any():
                    keep_mask[empty, self.sid_anchor_digits[0]] = 1.0
            correct_bias = self.sid_anchor_bias_from_logits(
                sid_anchor_logits,
                shuffle=self.sid_anchor_shuffle_condition,
                keep_mask=keep_mask,
            )
            base_logits = outputs.logits
            outputs.logits = base_logits.clone()
            outputs.logits[:, -self.tokenizer.n_digit:, :] += (
                self.sid_anchor_scale_parameter * correct_bias)
            if (self.training
                    and self.sid_anchor_counterfactual_weight > 0.0):
                wrong_bias = self.sid_anchor_bias_from_logits(
                    sid_anchor_logits,
                    shuffle=True,
                    keep_mask=keep_mask,
                )
                sid_anchor_wrong_target_logits = (
                    base_logits[:, -self.tokenizer.n_digit:, :]
                    + self.sid_anchor_scale_parameter * wrong_bias
                )

        logits = outputs.logits
        logits = logits[:, -masked_label_tokens.shape[1]:, :]

        logits = logits / self.temperature

        label_tokens = label_tokens.masked_fill(~label_attention_mask,
                                                self.tokenizer.ignored_label)

        token_loss = self.loss_fct(
            logits[masked_indices],
            label_tokens[masked_indices]) / p_mask[masked_indices]
        loss = token_loss.sum() / (label_tokens.shape[0] *
                                   label_tokens.shape[1])
        if (sid_anchor_logits is not None
                and self.sid_anchor_router_aux_weight > 0.0):
            router_losses = []
            for digit, digit_logits in enumerate(sid_anchor_logits):
                start = 1 + 256 * digit
                target_ids = (clean_label_tokens[:, digit] - start).clamp(
                    min=0, max=255)
                router_losses.append(F.cross_entropy(
                    digit_logits,
                    target_ids,
                    label_smoothing=0.05,
                ))
            sid_anchor_router_loss = torch.stack(router_losses).mean()
            loss = (
                loss
                + self.sid_anchor_router_aux_weight
                * sid_anchor_router_loss
            )
            outputs.sid_anchor_router_loss = sid_anchor_router_loss
        if (sid_anchor_logits is not None
                and self.sid_anchor_item_contrastive_weight > 0.0):
            anchor_target_ids = []
            contrastive_scores = None
            for digit in self.sid_anchor_digits:
                start = 1 + 256 * digit
                target_ids = (clean_label_tokens[:, digit] - start).clamp(
                    min=0, max=255)
                anchor_target_ids.append(target_ids)
                digit_logp = F.log_softmax(
                    sid_anchor_logits[digit], dim=-1)
                digit_scores = digit_logp[:, target_ids]
                contrastive_scores = (
                    digit_scores
                    if contrastive_scores is None
                    else contrastive_scores + digit_scores
                )
            contrastive_scores = (
                contrastive_scores
                / max(self.sid_anchor_item_contrastive_temperature, 1e-6)
            )
            target_signatures = torch.stack(
                anchor_target_ids, dim=-1)
            positive_mask = (
                target_signatures[:, None, :]
                == target_signatures[None, :, :]
            ).all(dim=-1)
            positive_log_mass = torch.logsumexp(
                contrastive_scores.masked_fill(~positive_mask, -torch.inf),
                dim=-1,
            )
            all_log_mass = torch.logsumexp(
                contrastive_scores, dim=-1)
            sid_anchor_item_contrastive_loss = (
                all_log_mass - positive_log_mass).mean()
            loss = (
                loss
                + self.sid_anchor_item_contrastive_weight
                * sid_anchor_item_contrastive_loss
            )
            outputs.sid_anchor_item_contrastive_loss = (
                sid_anchor_item_contrastive_loss)
        if (sid_anchor_teacher_logits is not None
                and self.sid_anchor_teacher_consistency_weight > 0.0):
            consistency_losses = []
            for student_logits, teacher_logits in zip(
                    sid_anchor_logits, sid_anchor_teacher_logits):
                consistency_losses.append(F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction='batchmean',
                ))
            sid_anchor_consistency_loss = torch.stack(
                consistency_losses).mean()
            loss = (
                loss
                + self.sid_anchor_teacher_consistency_weight
                * sid_anchor_consistency_loss
            )
            outputs.sid_anchor_consistency_loss = (
                sid_anchor_consistency_loss)
        if sid_anchor_wrong_target_logits is not None:
            correct_logits = outputs.logits[
                :, -masked_label_tokens.shape[1]:, :] / self.temperature
            wrong_logits = (
                sid_anchor_wrong_target_logits / self.temperature)
            safe_labels = label_tokens.clamp_min(0)
            correct_logp = F.log_softmax(
                correct_logits, dim=-1).gather(
                    -1, safe_labels.unsqueeze(-1)).squeeze(-1)
            wrong_logp = F.log_softmax(
                wrong_logits, dim=-1).gather(
                    -1, safe_labels.unsqueeze(-1)).squeeze(-1)
            weights = masked_indices.to(correct_logp.dtype)
            denominator = weights.sum(dim=-1).clamp_min(1.0)
            correct_score = (correct_logp * weights).sum(
                dim=-1) / denominator
            wrong_score = (wrong_logp * weights).sum(
                dim=-1) / denominator
            sid_anchor_counterfactual_loss = F.relu(
                self.sid_anchor_counterfactual_margin
                - correct_score
                + wrong_score
            ).mean()
            loss = (
                loss
                + self.sid_anchor_counterfactual_weight
                * sid_anchor_counterfactual_loss
            )
            outputs.sid_anchor_counterfactual_loss = (
                sid_anchor_counterfactual_loss)
        if (self.use_probabilistic_intent_memory
                and self.prob_intent_router_aux_weight > 0.0
                and outputs.prob_intent_distribution is not None):
            target_categories = self.prob_intent_item_categories[
                label_item_ids.long()]
            router_loss = F.nll_loss(
                outputs.prob_intent_distribution.clamp_min(1e-8).log(),
                target_categories,
            )
            loss = loss + self.prob_intent_router_aux_weight * router_loss
            outputs.probabilistic_intent_router_loss = router_loss
        if (self.use_probabilistic_intent_memory
                and self.prob_intent_counterfactual_weight > 0.0):
            wrong_outputs = self.llada_forward(
                input_ids=final_inputs_ids,
                attention_mask=final_attention_mask.long(),
                user_ids=user_ids,
                history_item_ids=history_item_ids,
                prob_intent_shuffle_override=True,
            )
            correct_logits = outputs.logits[
                :, -masked_label_tokens.shape[1]:, :] / self.temperature
            wrong_logits = wrong_outputs.logits[
                :, -masked_label_tokens.shape[1]:, :] / self.temperature
            safe_labels = label_tokens.clamp_min(0)
            correct_logp = F.log_softmax(
                correct_logits, dim=-1).gather(
                    -1, safe_labels.unsqueeze(-1)).squeeze(-1)
            wrong_logp = F.log_softmax(
                wrong_logits, dim=-1).gather(
                    -1, safe_labels.unsqueeze(-1)).squeeze(-1)
            weights = masked_indices.to(correct_logp.dtype)
            denominator = weights.sum(dim=-1).clamp_min(1.0)
            correct_score = (correct_logp * weights).sum(
                dim=-1) / denominator
            wrong_score = (wrong_logp * weights).sum(
                dim=-1) / denominator
            counterfactual_loss = F.relu(
                self.prob_intent_counterfactual_margin
                - correct_score
                + wrong_score
            ).mean()
            loss = (
                loss
                + self.prob_intent_counterfactual_weight
                * counterfactual_loss
            )
            outputs.probabilistic_intent_counterfactual_loss = (
                counterfactual_loss
            )
        if (self.use_collaborative_intent_memory
                and self.collab_intent_counterfactual_weight > 0.0):
            wrong_outputs = self.llada_forward(
                input_ids=final_inputs_ids,
                attention_mask=final_attention_mask.long(),
                user_ids=user_ids,
                history_item_ids=history_item_ids,
                collab_intent_shuffle_override=True,
            )
            correct_logits = outputs.logits[
                :, -masked_label_tokens.shape[1]:, :] / self.temperature
            wrong_logits = wrong_outputs.logits[
                :, -masked_label_tokens.shape[1]:, :] / self.temperature
            safe_labels = label_tokens.clamp_min(0)
            correct_logp = F.log_softmax(
                correct_logits, dim=-1).gather(
                    -1, safe_labels.unsqueeze(-1)).squeeze(-1)
            wrong_logp = F.log_softmax(
                wrong_logits, dim=-1).gather(
                    -1, safe_labels.unsqueeze(-1)).squeeze(-1)
            weights = masked_indices.to(correct_logp.dtype)
            denominator = weights.sum(dim=-1).clamp_min(1.0)
            correct_score = (correct_logp * weights).sum(
                dim=-1) / denominator
            wrong_score = (wrong_logp * weights).sum(
                dim=-1) / denominator
            counterfactual_loss = F.relu(
                self.collab_intent_counterfactual_margin
                - correct_score
                + wrong_score
            ).mean()
            loss = (
                loss
                + self.collab_intent_counterfactual_weight
                * counterfactual_loss
            )
            outputs.collaborative_intent_counterfactual_loss = (
                counterfactual_loss
            )
        if self.use_semantic_bridge:
            if label_item_ids is None:
                raise ValueError(
                    'Semantic-Bridge loss requires label item IDs')
            semantic_loss = self.semantic_bridge_alignment_loss(
                outputs,
                masked_indices,
                label_item_ids,
            )
            loss = loss + self.semantic_bridge_weight * semantic_loss
            outputs.semantic_bridge_loss = semantic_loss
        if self.use_multi_intent_memory:
            if label_item_ids is None or history_item_ids is None:
                raise ValueError(
                    'Multi-Intent Memory loss requires history and label '
                    'item IDs')
            if (outputs.multi_intent_slots is None
                    or outputs.multi_intent_probabilities is None):
                raise ValueError(
                    'Multi-Intent Memory was not built during target denoising'
                )
            multi_intent_loss = self.multi_intent_memory_alignment_loss(
                outputs.multi_intent_slots,
                outputs.multi_intent_probabilities,
                label_item_ids,
            )
            loss = loss + self.multi_intent_recon_weight * multi_intent_loss
            outputs.multi_intent_memory_loss = multi_intent_loss
        if self.use_dialogue_multi_intent:
            if label_item_ids is None:
                raise ValueError(
                    'Dialogue multi-intent loss requires label item IDs')
            if (outputs.dialogue_multi_intent_slots is None
                    or outputs.dialogue_multi_intent_probabilities is None):
                raise ValueError(
                    'Dialogue multi-intent control was not built during '
                    'target denoising')
            dialogue_multi_intent_loss = (
                self.dialogue_multi_intent_alignment_loss(
                    outputs.dialogue_multi_intent_slots,
                    outputs.dialogue_multi_intent_probabilities,
                    label_item_ids,
                )
            )
            loss = (
                loss
                + self.dialogue_multi_intent_recon_weight
                * dialogue_multi_intent_loss
            )
            outputs.multi_intent_memory_loss = dialogue_multi_intent_loss
        if self.use_qwen_teacher_distill and user_ids is not None:
            qwen_loss = self.qwen_teacher_distill_loss(
                logits=logits,
                masked_indices=masked_indices,
                p_mask=p_mask,
                user_ids=user_ids,
            )
            loss = loss + self.qwen_distill_weight * qwen_loss
            outputs.qwen_distill_loss = qwen_loss
        outputs.loss = loss

        return outputs

    def qwen_teacher_distill_loss(
        self,
        logits: torch.Tensor,
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_ids = user_ids.long().clamp(
            min=0, max=self.qwen_teacher_item_ids.shape[0] - 1)
        teacher_items = self.qwen_teacher_item_ids[user_ids]
        teacher_scores = self.qwen_teacher_scores[user_ids]
        valid_items = teacher_items > 0
        covered = valid_items.any(dim=1)
        if not covered.any():
            return logits.new_zeros(())

        teacher_items = teacher_items.clamp(min=0,
                                            max=self.item_id2tokens.shape[0] -
                                            1)
        teacher_tokens = self.item_id2tokens[teacher_items]
        weights = teacher_scores.masked_fill(~valid_items, -1e4)
        weights = F.softmax(weights, dim=-1)

        batch_size, n_digit, vocab_size = logits.shape
        teacher_probs = logits.new_zeros((batch_size, n_digit, vocab_size))
        for pos in range(n_digit):
            token_ids = teacher_tokens[:, :, pos]
            pos_probs = logits.new_zeros((batch_size, vocab_size))
            pos_probs.scatter_add_(1, token_ids, weights)
            teacher_probs[:, pos, :] = pos_probs

        teacher_probs = teacher_probs.clamp_min(1e-8)
        teacher_probs = teacher_probs / teacher_probs.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        active = masked_indices & covered.unsqueeze(1)
        if not active.any():
            return logits.new_zeros(())

        student_logp = F.log_softmax(
            logits / self.qwen_distill_temperature, dim=-1)
        kl = F.kl_div(student_logp[active],
                      teacher_probs[active],
                      reduction='none').sum(dim=-1)
        kl = kl / p_mask[active].clamp_min(1e-6)
        return kl.sum() / (logits.shape[0] * logits.shape[1])

    def user_code_mask_recon_loss(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        clean_embeds = self.add_item_pos_emb(input_tokens)
        with torch.no_grad():
            teacher_context, teacher_logits, _ = self.get_user_code_context(
                input_tokens, clean_embeds, return_aux=True)
            teacher_probs = F.softmax(
                teacher_logits / self.user_code_mask_temperature, dim=-1)

        random_mask = torch.rand_like(input_tokens,
                                      dtype=torch.float) < self.user_code_mask_ratio
        random_mask = random_mask & input_attention_mask
        masked_tokens = torch.where(random_mask, self.mask_token_id,
                                    input_tokens)
        masked_embeds = self.add_item_pos_emb(masked_tokens)
        student_context, student_logits, _ = self.get_user_code_context(
            masked_tokens, masked_embeds, return_aux=True)

        student_logp = F.log_softmax(
            student_logits / self.user_code_mask_temperature, dim=-1)
        kl = F.kl_div(student_logp,
                      teacher_probs,
                      reduction='batchmean') * (
                          self.user_code_mask_temperature**2)
        context_loss = F.mse_loss(student_context,
                                  teacher_context.detach(),
                                  reduction='mean')
        return kl + self.user_code_mask_context_weight * context_loss

    def user_code_aux_recon_loss(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
        label_tokens: torch.Tensor,
        label_attention_mask: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.add_item_pos_emb(input_tokens)
        code_context = self.get_user_code_context(input_tokens, input_embeds)
        logits = self.user_code_logits_bias(code_context)
        logits = logits[:, None, :].expand(-1, label_tokens.shape[1], -1)
        logits = logits / self.user_code_aux_temperature

        labels = label_tokens.masked_fill(~label_attention_mask,
                                          self.tokenizer.ignored_label)
        ce = self.loss_fct(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1))
        ce = ce.reshape_as(label_tokens)
        ce = ce.sum() / label_attention_mask.sum().clamp_min(1)

        loss = ce
        if self.user_code_aux_teacher_weight > 0.0 and user_ids is not None:
            masked_indices = label_attention_mask.bool()
            p_mask = torch.ones_like(label_tokens, dtype=logits.dtype)
            teacher_loss = self.qwen_teacher_distill_loss(
                logits=logits,
                masked_indices=masked_indices,
                p_mask=p_mask,
                user_ids=user_ids,
            )
            loss = loss + self.user_code_aux_teacher_weight * teacher_loss
        return loss

    def rec_control_condition_alignment_loss(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
        label_tokens: torch.Tensor,
        label_attention_mask: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.add_item_pos_emb(input_tokens)
        code_context = None
        reason_context = None
        if self.use_user_codebook:
            code_context = self.get_user_code_context(input_tokens,
                                                      input_embeds)
        if self.use_qwen_reasoning_adapter:
            reason_context = self.get_qwen_reasoning_context(user_ids)
        condition_context = self.get_rec_control_condition(
            code_context, reason_context)
        if condition_context is None:
            return input_embeds.new_zeros(())

        condition_context = self.rec_control_align_proj(condition_context)
        label_embeds = self.llada.model.transformer.wte(label_tokens)
        label_mask = label_attention_mask.unsqueeze(-1).to(label_embeds.dtype)
        label_context = (label_embeds * label_mask).sum(dim=1)
        label_context = label_context / label_mask.sum(dim=1).clamp_min(1.0)

        condition_context = F.normalize(condition_context, dim=-1)
        label_context = F.normalize(label_context.detach(), dim=-1)
        logits = torch.matmul(condition_context,
                              label_context.t()) / self.rec_control_align_temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)

    def get_item_sid_context(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.long().clamp(
            min=0, max=self.item_id2tokens.shape[0] - 1)
        item_tokens = self.item_id2tokens[item_ids]
        item_embeds = self.llada.model.transformer.wte(item_tokens)
        return item_embeds.mean(dim=1)

    def rec_control_item_rank_loss(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
        label_item_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.add_item_pos_emb(input_tokens)
        code_context = None
        reason_context = None
        if self.use_user_codebook:
            code_context = self.get_user_code_context(input_tokens,
                                                      input_embeds)
        if self.use_qwen_reasoning_adapter:
            reason_context = self.get_qwen_reasoning_context(user_ids)
        condition_context = self.get_rec_control_condition(
            code_context, reason_context)
        if condition_context is None:
            return input_embeds.new_zeros(())

        condition_context = self.rec_control_item_rank_proj(condition_context)
        condition_context = F.normalize(condition_context, dim=-1)
        if self.rec_control_item_rank_use_sampled:
            loss = self.rec_control_sampled_item_rank_loss(
                condition_context=condition_context,
                label_item_ids=label_item_ids,
                user_ids=user_ids,
            )
            if self.rec_control_item_rank_inbatch_weight > 0.0:
                inbatch_loss = self.rec_control_inbatch_item_rank_loss(
                    condition_context=condition_context,
                    label_item_ids=label_item_ids,
                    user_ids=user_ids,
                )
                loss = loss + self.rec_control_item_rank_inbatch_weight * \
                    inbatch_loss
            return loss

        return self.rec_control_inbatch_item_rank_loss(
            condition_context=condition_context,
            label_item_ids=label_item_ids,
            user_ids=user_ids,
        )

    def rec_control_sid_token_loss(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
        label_tokens: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.add_item_pos_emb(input_tokens)
        code_context = None
        reason_context = None
        if self.use_user_codebook:
            code_context = self.get_user_code_context(input_tokens,
                                                      input_embeds)
        if self.use_qwen_reasoning_adapter:
            reason_context = self.get_qwen_reasoning_context(user_ids)
        condition_context = self.get_rec_control_condition(
            code_context, reason_context)
        if condition_context is None:
            return input_embeds.new_zeros(())

        batch_size = label_tokens.shape[0]
        n_digit = self.tokenizer.n_digit
        vocab_size = self.tokenizer.vocab_size
        logits = self.rec_control_sid_token_proj(condition_context)
        logits = logits.view(batch_size, n_digit, vocab_size)
        logits = logits / self.rec_control_sid_token_temperature

        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            label_tokens.reshape(-1),
            reduction='none',
        ).view(batch_size, n_digit)
        if self.rec_control_sid_token_prefix_decay != 1.0:
            pos = torch.arange(n_digit,
                               device=label_tokens.device,
                               dtype=token_loss.dtype)
            weights = self.rec_control_sid_token_prefix_decay**pos
            weights = weights / weights.mean().clamp_min(1e-8)
            token_loss = token_loss * weights.unsqueeze(0)
        loss = token_loss.mean()

        if (self.rec_control_sid_token_teacher_weight > 0.0
                and self.use_qwen_teacher_distill and user_ids is not None):
            user_ids = user_ids.long().clamp(
                min=0, max=self.qwen_teacher_item_ids.shape[0] - 1)
            teacher_items = self.qwen_teacher_item_ids[user_ids]
            teacher_scores = self.qwen_teacher_scores[user_ids]
            valid_items = teacher_items > 0
            teacher_items = teacher_items.clamp(
                min=0, max=self.item_id2tokens.shape[0] - 1)
            teacher_tokens = self.item_id2tokens[teacher_items]
            weights = teacher_scores.masked_fill(~valid_items, -1e4)
            weights = F.softmax(weights, dim=1)
            weights = weights * valid_items.to(weights.dtype)
            weights = weights / weights.sum(dim=1,
                                            keepdim=True).clamp_min(1e-8)
            teacher_probs = logits.new_zeros((batch_size, n_digit, vocab_size))
            for pos in range(n_digit):
                teacher_probs[:, pos, :].scatter_add_(
                    dim=1,
                    index=teacher_tokens[:, :, pos],
                    src=weights,
                )
            teacher_probs = teacher_probs.clamp_min(1e-8)
            teacher_probs = teacher_probs / teacher_probs.sum(
                dim=-1, keepdim=True).clamp_min(1e-8)
            teacher_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                teacher_probs,
                reduction='batchmean',
            )
            loss = loss + self.rec_control_sid_token_teacher_weight * \
                teacher_loss
        return loss

    def rec_control_inbatch_item_rank_loss(
        self,
        condition_context: torch.Tensor,
        label_item_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        item_context = self.get_item_sid_context(label_item_ids).detach()
        item_context = F.normalize(item_context, dim=-1)
        logits = torch.matmul(
            condition_context,
            item_context.t()) / self.rec_control_item_rank_temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, targets)

        if (self.rec_control_item_rank_teacher_weight > 0.0
                and self.use_qwen_teacher_distill and user_ids is not None):
            user_ids = user_ids.long().clamp(
                min=0, max=self.qwen_teacher_item_ids.shape[0] - 1)
            teacher_items = self.qwen_teacher_item_ids[user_ids]
            teacher_scores = self.qwen_teacher_scores[user_ids]
            valid_items = teacher_items > 0
            teacher_items = teacher_items.clamp(
                min=0, max=self.item_id2tokens.shape[0] - 1)

            matches = teacher_items[:, :, None] == label_item_ids[None, None, :]
            teacher_targets = (matches.to(logits.dtype) *
                               teacher_scores.masked_fill(
                                   ~valid_items, -1e4).softmax(dim=1)
                               [:, :, None]).sum(dim=1)
            teacher_targets.scatter_add_(
                1, targets[:, None],
                torch.ones_like(targets, dtype=logits.dtype)[:, None])
            teacher_targets = teacher_targets / teacher_targets.sum(
                dim=1, keepdim=True).clamp_min(1e-8)
            teacher_kl = F.kl_div(F.log_softmax(logits, dim=-1),
                                  teacher_targets,
                                  reduction='batchmean')
            loss = loss + self.rec_control_item_rank_teacher_weight * teacher_kl
        return loss

    def rec_control_sampled_item_rank_loss(
        self,
        condition_context: torch.Tensor,
        label_item_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        label_item_ids = label_item_ids.long().view(-1)
        batch_size = label_item_ids.shape[0]
        device = label_item_ids.device

        candidate_chunks = [label_item_ids[:, None]]
        teacher_target = None
        if (self.rec_control_item_rank_teacher_weight > 0.0
                and self.use_qwen_teacher_distill and user_ids is not None
                and self.rec_control_item_rank_num_teacher > 0):
            user_ids = user_ids.long().clamp(
                min=0, max=self.qwen_teacher_item_ids.shape[0] - 1)
            num_teacher = min(self.rec_control_item_rank_num_teacher,
                              self.qwen_teacher_item_ids.shape[1])
            teacher_items = self.qwen_teacher_item_ids[user_ids, :num_teacher]
            teacher_scores = self.qwen_teacher_scores[user_ids, :num_teacher]
            teacher_valid = teacher_items > 0
            teacher_items = teacher_items.clamp(
                min=0, max=self.item_id2tokens.shape[0] - 1)
            candidate_chunks.append(teacher_items)
            masked_scores = teacher_scores.masked_fill(~teacher_valid, -1e4)
            teacher_target = masked_scores.softmax(dim=1)
            teacher_target = teacher_target * teacher_valid.to(
                teacher_target.dtype)
            teacher_target = teacher_target / teacher_target.sum(
                dim=1, keepdim=True).clamp_min(1e-8)

        if self.rec_control_item_rank_num_random > 0:
            random_items = torch.randint(
                low=1,
                high=max(2, self.item_id2tokens.shape[0]),
                size=(batch_size, self.rec_control_item_rank_num_random),
                device=device,
            )
            candidate_chunks.append(random_items)

        candidate_ids = torch.cat(candidate_chunks, dim=1)
        flat_context = self.get_item_sid_context(
            candidate_ids.reshape(-1)).detach()
        candidate_context = flat_context.reshape(batch_size,
                                                 candidate_ids.shape[1], -1)
        candidate_context = F.normalize(candidate_context, dim=-1)
        logits = torch.einsum(
            'bd,bcd->bc', condition_context,
            candidate_context) / self.rec_control_item_rank_temperature

        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets)

        if teacher_target is not None:
            soft_target = logits.new_zeros(logits.shape)
            soft_target[:, 0] = self.rec_control_item_rank_label_weight
            teacher_start = 1
            teacher_end = teacher_start + teacher_target.shape[1]
            soft_target[:, teacher_start:teacher_end] += \
                self.rec_control_item_rank_teacher_weight * teacher_target
            soft_target = soft_target / soft_target.sum(
                dim=1, keepdim=True).clamp_min(1e-8)
            teacher_loss = F.kl_div(F.log_softmax(logits, dim=-1),
                                    soft_target,
                                    reduction='batchmean')
            loss = loss + self.rec_control_item_rank_teacher_weight * \
                teacher_loss
        return loss

    def rec_control_pretrain_loss(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
        label_tokens: torch.Tensor,
        label_attention_mask: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        masked_label_tokens = torch.where(
            label_attention_mask,
            torch.full_like(label_tokens, self.mask_token_id),
            label_tokens,
        )
        final_inputs_ids = torch.cat([input_tokens, masked_label_tokens],
                                     dim=1)
        final_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)

        outputs = self.llada_forward(
            input_ids=final_inputs_ids,
            attention_mask=final_attention_mask.long(),
            user_ids=user_ids,
        )
        logits = outputs.logits[:, -label_tokens.shape[1]:, :]
        logits = logits / self.temperature
        labels = label_tokens.masked_fill(~label_attention_mask,
                                          self.tokenizer.ignored_label)
        ce = self.loss_fct(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1))
        ce = ce.reshape_as(label_tokens)
        recon_loss = ce.sum() / label_attention_mask.sum().clamp_min(1)

        if (self.use_qwen_teacher_distill
                and self.rec_control_pretrain_qwen_weight > 0.0
                and user_ids is not None):
            masked_indices = label_attention_mask.bool()
            p_mask = torch.ones_like(label_tokens, dtype=logits.dtype)
            qwen_loss = self.qwen_teacher_distill_loss(
                logits=logits,
                masked_indices=masked_indices,
                p_mask=p_mask,
                user_ids=user_ids,
            )
            recon_loss = recon_loss + \
                self.rec_control_pretrain_qwen_weight * qwen_loss

        align_loss = None
        if self.rec_control_align_weight > 0.0:
            align_loss = self.rec_control_condition_alignment_loss(
                input_tokens=input_tokens,
                input_attention_mask=input_attention_mask,
                label_tokens=label_tokens,
                label_attention_mask=label_attention_mask,
                user_ids=user_ids,
            )
            recon_loss = recon_loss + self.rec_control_align_weight * align_loss
        return recon_loss, align_loss

    def forward(self, batch: dict):
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_tokens = input_tokens.reshape((input_tokens.shape[0], -1))
        input_attention_mask = input_tokens != 0

        assert 'labels' in batch, 'The batch must contain the labels.'
        label_tokens = self.item_id2tokens[batch['labels'].squeeze(1)]
        label_tokens = label_tokens.reshape((label_tokens.shape[0], -1))
        label_attention_mask = torch.ones_like(
            label_tokens, device=label_tokens.device).bool()

        rec_control_pretrain_loss = None
        rec_control_align_loss = None
        rec_control_item_rank_loss = None
        rec_control_sid_token_loss = None
        if self.use_rec_control_pretrain:
            rec_control_pretrain_loss, rec_control_align_loss = \
                self.rec_control_pretrain_loss(
                    input_tokens=input_tokens,
                    input_attention_mask=input_attention_mask,
                    label_tokens=label_tokens,
                    label_attention_mask=label_attention_mask,
                    user_ids=batch.get('user_ids'),
                )
            if self.rec_control_pretrain_only:
                zero = rec_control_pretrain_loss.new_zeros(())
                return LLaDARecOutPut(
                    loss=self.rec_control_pretrain_weight *
                    rec_control_pretrain_loss,
                    his_mask_loss=zero,
                    target_mask_loss=zero,
                    rec_control_pretrain_loss=rec_control_pretrain_loss,
                    rec_control_align_loss=rec_control_align_loss,
                )
        if self.use_rec_control_item_rank:
            rec_control_item_rank_loss = self.rec_control_item_rank_loss(
                input_tokens=input_tokens,
                input_attention_mask=input_attention_mask,
                label_item_ids=batch['labels'].squeeze(1),
                user_ids=batch.get('user_ids'),
            )
        if self.use_rec_control_sid_token:
            rec_control_sid_token_loss = self.rec_control_sid_token_loss(
                input_tokens=input_tokens,
                input_attention_mask=input_attention_mask,
                label_tokens=label_tokens,
                user_ids=batch.get('user_ids'),
            )

        user_code_mask_loss = None
        if self.use_user_code_mask_recon:
            user_code_mask_loss = self.user_code_mask_recon_loss(
                input_tokens=input_tokens,
                input_attention_mask=input_attention_mask,
            )
            if self.user_code_mask_only:
                zero = user_code_mask_loss.new_zeros(())
                return LLaDARecOutPut(
                    loss=self.user_code_mask_weight * user_code_mask_loss,
                    his_mask_loss=zero,
                    target_mask_loss=zero,
                    user_code_mask_loss=user_code_mask_loss,
                )

        aux_loss = None
        if self.use_user_code_aux_recon:
            aux_loss = self.user_code_aux_recon_loss(
                input_tokens=input_tokens,
                input_attention_mask=input_attention_mask,
                label_tokens=label_tokens,
                label_attention_mask=label_attention_mask,
                user_ids=batch.get('user_ids'),
            )
            if self.user_code_aux_only:
                zero = aux_loss.new_zeros(())
                return LLaDARecOutPut(
                    loss=self.user_code_aux_weight * aux_loss,
                    his_mask_loss=zero,
                    target_mask_loss=zero,
                    user_code_aux_loss=aux_loss,
                )

        his_inputs = torch.cat([input_tokens, label_tokens], dim=1)
        his_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)
        his_outputs = self.his_mask_loss(his_inputs, his_attention_mask)
        his_loss = his_outputs.loss

        target_outputs = self.target_mask_loss(input_tokens,
                                               input_attention_mask,
                                               label_tokens,
                                               label_attention_mask,
                                               batch.get('user_ids'),
                                               batch['labels'].squeeze(1),
                                               batch['input_ids'])
        target_loss = target_outputs.loss

        loss = target_loss + self.his_mask_w * his_loss
        if aux_loss is not None and self.user_code_aux_weight > 0.0:
            loss = loss + self.user_code_aux_weight * aux_loss
        if (user_code_mask_loss is not None
                and self.user_code_mask_weight > 0.0):
            loss = loss + self.user_code_mask_weight * user_code_mask_loss
        if (rec_control_pretrain_loss is not None
                and self.rec_control_pretrain_weight > 0.0):
            loss = loss + \
                self.rec_control_pretrain_weight * rec_control_pretrain_loss
        if (rec_control_item_rank_loss is not None
                and self.rec_control_item_rank_weight > 0.0):
            loss = loss + \
                self.rec_control_item_rank_weight * rec_control_item_rank_loss
        if (rec_control_sid_token_loss is not None
                and self.rec_control_sid_token_weight > 0.0):
            loss = loss + \
                self.rec_control_sid_token_weight * rec_control_sid_token_loss

        return LLaDARecOutPut(loss=loss,
                              his_mask_loss=his_loss,
                              target_mask_loss=target_loss,
                              qwen_distill_loss=getattr(
                                  target_outputs, 'qwen_distill_loss', None),
                              user_code_aux_loss=aux_loss,
                              user_code_mask_loss=user_code_mask_loss,
                              rec_control_pretrain_loss=rec_control_pretrain_loss,
                              rec_control_align_loss=rec_control_align_loss,
                              rec_control_item_rank_loss=rec_control_item_rank_loss,
                              rec_control_sid_token_loss=rec_control_sid_token_loss,
                              semantic_bridge_loss=getattr(
                                  target_outputs,
                                  'semantic_bridge_loss',
                                  None),
                              multi_intent_memory_loss=getattr(
                                  target_outputs,
                                  'multi_intent_memory_loss',
                                  None),
                              probabilistic_intent_counterfactual_loss=getattr(
                                  target_outputs,
                                  'probabilistic_intent_counterfactual_loss',
                                  None),
                              probabilistic_intent_router_loss=getattr(
                                  target_outputs,
                                  'probabilistic_intent_router_loss',
                                  None),
                              collaborative_intent_counterfactual_loss=getattr(
                                  target_outputs,
                                  'collaborative_intent_counterfactual_loss',
                                  None),
                              sid_anchor_router_loss=getattr(
                                  target_outputs,
                                  'sid_anchor_router_loss',
                                  None),
                              sid_anchor_counterfactual_loss=getattr(
                                  target_outputs,
                                  'sid_anchor_counterfactual_loss',
                                  None),
                              sid_anchor_consistency_loss=getattr(
                                  target_outputs,
                                  'sid_anchor_consistency_loss',
                                  None),
                              sid_anchor_item_contrastive_loss=getattr(
                                  target_outputs,
                                  'sid_anchor_item_contrastive_loss',
                                  None))

    def get_masked_seqs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_target:
            return self.get_masked_targets(input_ids, attention_mask)
        else:
            return self.get_masked_seqs_his(input_ids, attention_mask)

    def get_masked_seqs_his(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape

        masked_num = torch.randint(low=0,
                                   high=self.tokenizer.n_digit + 1,
                                   size=(b, l // self.tokenizer.n_digit),
                                   device=input_ids.device)

        masked_prob = masked_num / self.tokenizer.n_digit
        masked_prob = torch.repeat_interleave(masked_prob,
                                              self.tokenizer.n_digit,
                                              dim=1)

        masked_indices = torch.rand((b, l),
                                    device=input_ids.device) < masked_prob

        noisy_batch = torch.where(masked_indices, self.mask_token_id,
                                  input_ids)

        noisy_batch = noisy_batch.masked_fill(~attention_mask, 0)
        masked_indices = masked_indices.masked_fill(~attention_mask, False)

        return noisy_batch, masked_indices, masked_prob

    def get_masked_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape

        mask_num = torch.randint(1,
                                 self.tokenizer.n_digit + 1, (b, ),
                                 device=input_ids.device)
        if self.training and self.target_full_mask_prob > 0.0:
            full_mask = torch.rand(
                b, device=input_ids.device) < self.target_full_mask_prob
            mask_num = torch.where(
                full_mask,
                torch.full_like(mask_num, self.tokenizer.n_digit),
                mask_num)
        p_mask = mask_num / self.tokenizer.n_digit

        rand_scores = torch.rand((b, l), device=input_ids.device)
        rand_scores = rand_scores.masked_fill(attention_mask == 0, 2.0)
        sorted_idx = torch.argsort(rand_scores, dim=1)  # (b, l)

        sort_input_ids = input_ids.clone()
        sort_input_ids = sort_input_ids[torch.arange(b).unsqueeze(1),
                                        sorted_idx]
        sort_mask_pos = torch.arange(
            l, device=input_ids.device)[None, :] < mask_num[:, None]
        sort_input_ids[sort_mask_pos] = self.mask_token_id

        masked_input_ids = input_ids.clone()
        masked_input_ids[torch.arange(b).unsqueeze(1),
                         sorted_idx] = sort_input_ids

        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[masked_input_ids == self.mask_token_id] = True

        return masked_input_ids, mask_positions, p_mask[:, None].repeat(1, l)

    def get_transfer_index(self, input_ids, logits, num_transfer_tokens):
        mask_index = (input_ids == self.mask_token_id)

        x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = p.gather(index=x0.unsqueeze(-1), dim=-1).squeeze(-1)

        x0 = torch.where(mask_index, x0, input_ids)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        _, transfer_index = torch.topk(confidence,
                                       k=num_transfer_tokens,
                                       dim=-1)

        return transfer_index

    def generate(self, batch, n_return_sequences=1, return_scores=False):
        if batch['split'] == 'val':
            outputs = self.beam_search(batch,
                                       n_return_sequences,
                                       num_beams=self.val_num_beams,
                                       return_scores=return_scores)
        else:
            outputs = self.beam_search(batch,
                                       n_return_sequences,
                                       num_beams=self.num_beams,
                                       return_scores=return_scores)

        return outputs

    def get_sid_anchor_logits(
        self,
        history_item_ids: torch.Tensor,
    ) -> Optional[list[torch.Tensor]]:
        if self.sid_anchor_router is None:
            return None
        if self.training and self.sid_anchor_trainable_router:
            return self.sid_anchor_router(history_item_ids.long())
        self.sid_anchor_router.eval()
        with torch.no_grad():
            return self.sid_anchor_router(history_item_ids.long())

    def sid_anchor_bias_from_logits(
        self,
        logits: list[torch.Tensor],
        shuffle: bool = False,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = logits[0].shape[0]
        bias = logits[0].new_zeros(
            (
                batch_size,
                self.tokenizer.n_digit,
                self.tokenizer.vocab_size,
            )
        )
        for digit in self.sid_anchor_digits:
            log_probability = F.log_softmax(logits[digit], dim=-1)
            if shuffle:
                log_probability = torch.roll(
                    log_probability, shifts=137, dims=-1)
            log_probability = (
                log_probability
                - log_probability.mean(dim=-1, keepdim=True)
            )
            if keep_mask is not None:
                log_probability = (
                    log_probability * keep_mask[:, digit, None])
            start = 1 + 256 * digit
            bias[:, digit, start:start + 256] = log_probability
        return bias

    def build_sid_anchor_bias(
        self,
        history_item_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.get_sid_anchor_logits(history_item_ids)
        if logits is None:
            return None
        return self.sid_anchor_bias_from_logits(
            logits,
            shuffle=self.sid_anchor_shuffle_condition,
        )

    def build_dialogue_sid_bias(
        self,
        user_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Look up a target-safe dialogue posterior in local SID vocabularies."""
        if not self.use_dialogue_sid_condition or user_ids is None:
            return None
        user_ids = user_ids.long()
        in_range = (
            (user_ids >= 0)
            & (user_ids < self.dialogue_sid_user_lookup.shape[0])
        )
        safe_user_ids = user_ids.clamp(
            min=0, max=self.dialogue_sid_user_lookup.shape[0] - 1)
        rows = self.dialogue_sid_user_lookup[safe_user_ids]
        valid = in_range & (rows >= 0)
        safe_rows = rows.clamp(
            min=0, max=self.dialogue_local_sid_bias.shape[0] - 1)
        bias = self.dialogue_local_sid_bias[safe_rows]
        return bias * valid[:, None, None].to(bias.dtype)

    def build_dialogue_multi_intent_sid_bias(
        self,
        user_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (not self.use_dialogue_multi_intent_sid_guidance
                or user_ids is None):
            return None
        resolved_user_ids, probabilities = (
            self.resolve_dialogue_multi_intent_condition(user_ids)
        )
        per_slot_bias = self.dialogue_multi_intent_sid_bias[
            resolved_user_ids]
        return torch.einsum(
            'bk,bkdc->bdc', probabilities, per_slot_bias)

    def apply_dialogue_multi_intent_sid_bias(
        self,
        logits: torch.Tensor,
        local_sid_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if local_sid_bias is None:
            return logits
        logits = logits.clone()
        target_start = logits.shape[1] - self.tokenizer.n_digit
        vocabulary_start = 1
        for depth, codebook_size in enumerate(
                self.tokenizer.codebook_sizes):
            logits[
                :,
                target_start + depth,
                vocabulary_start:vocabulary_start + codebook_size,
            ] += (
                self.dialogue_multi_intent_sid_guidance_scale
                * local_sid_bias[:, depth, :codebook_size].to(logits.dtype)
            )
            vocabulary_start += codebook_size
        return logits

    def apply_dialogue_sid_bias(
        self,
        logits: torch.Tensor,
        local_sid_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if local_sid_bias is None:
            return logits
        logits = logits.clone()
        target_start = logits.shape[1] - self.tokenizer.n_digit
        local_sid_bias = local_sid_bias.to(logits.dtype)
        for depth in range(self.tokenizer.n_digit):
            vocabulary_start = 1 + 256 * depth
            logits[
                :,
                target_start + depth,
                vocabulary_start:vocabulary_start + 256,
            ] += self.dialogue_sid_condition_scale * local_sid_bias[:, depth]
        return logits

    def beam_search(self,
                    batch,
                    num_return_sequences=1,
                    num_beams=1,
                    return_scores=False):
        if self.use_sid_trie_beam:
            return self.beam_search_sid_trie(batch, num_return_sequences,
                                             num_beams, return_scores)

        assert num_beams >= num_return_sequences

        input_ids: torch.Tensor = self.item_id2tokens[batch['input_ids']]
        input_ids = input_ids.reshape((input_ids.shape[0], -1))
        attention_mask: torch.Tensor = input_ids != 0
        history_item_ids = batch['input_ids']
        sid_anchor_bias = self.build_sid_anchor_bias(history_item_ids)

        batch_size = input_ids.shape[0]
        n_digit = self.tokenizer.n_digit
        user_ids = batch.get('user_ids')
        dialogue_sid_bias = self.build_dialogue_sid_bias(user_ids)
        dialogue_intent_sid_bias = (
            self.build_dialogue_multi_intent_sid_bias(user_ids))
        if user_ids is not None:
            user_ids = user_ids.repeat_interleave(num_beams, dim=0)
        history_item_ids = history_item_ids.repeat_interleave(
            num_beams, dim=0)
        if sid_anchor_bias is not None:
            sid_anchor_bias = sid_anchor_bias.repeat_interleave(
                num_beams, dim=0)
        if dialogue_sid_bias is not None:
            dialogue_sid_bias = dialogue_sid_bias.repeat_interleave(
                num_beams, dim=0)
        if dialogue_intent_sid_bias is not None:
            dialogue_intent_sid_bias = (
                dialogue_intent_sid_bias.repeat_interleave(
                    num_beams, dim=0))

        # Prepare beam search inputs
        input_ids, attention_mask, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
            input_ids, attention_mask, batch_size, num_beams)

        num_transfer_tokens = n_digit // self.gen_steps
        num_transfered = 0
        for i in range(self.gen_steps):

            if i == (self.gen_steps - 1):
                num_transfer_tokens = n_digit - num_transfered

            outputs = self.llada_forward(input_ids=input_ids,
                                         attention_mask=attention_mask.long(),
                                         user_ids=user_ids,
                                         history_item_ids=history_item_ids)

            logits = outputs.logits
            if sid_anchor_bias is not None:
                logits = logits.clone()
                logits[:, -n_digit:, :] += (
                    self.sid_anchor_scale_parameter * sid_anchor_bias)
            logits = self.apply_dialogue_sid_bias(
                logits, dialogue_sid_bias)
            logits = self.apply_dialogue_multi_intent_sid_bias(
                logits, dialogue_intent_sid_bias)
            if self.user_condition_guidance_scale != 0.0:
                uncond_outputs = self.llada_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask.long(),
                    user_ids=user_ids,
                    history_item_ids=history_item_ids,
                    disable_user_conditions=True,
                )
                logits = uncond_outputs.logits + \
                    self.user_condition_guidance_scale * \
                    (logits - uncond_outputs.logits)
            logits = logits / self.temperature

            transfer_index = self.get_transfer_index(input_ids, logits,
                                                     num_transfer_tokens)

            for j in range(num_transfer_tokens):
                input_ids, beam_scores = self.beam_search_step(
                    logits, input_ids, transfer_index[:, j], beam_scores,
                    beam_idx_offset, batch_size, num_beams, user_ids)

            num_transfered += num_transfer_tokens

        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        label_ids = input_ids[:, -n_digit:]
        label_ids = label_ids[selection_mask.view(-1), :].reshape(
            -1, num_return_sequences, n_digit)

        if return_scores:
            selected_scores = beam_scores.reshape(batch_size, num_beams)[
                :, :num_return_sequences]
            return label_ids, selected_scores
        return label_ids

    def beam_search_sid_trie(self,
                             batch,
                             num_return_sequences=1,
                             num_beams=1,
                             return_scores=False):
        if not self.sid_trie_left_to_right:
            raise NotImplementedError(
                "SID trie beam currently supports left-to-right decoding only")
        assert num_beams >= num_return_sequences

        input_ids: torch.Tensor = self.item_id2tokens[batch['input_ids']]
        input_ids = input_ids.reshape((input_ids.shape[0], -1))
        attention_mask: torch.Tensor = input_ids != 0
        history_item_ids = batch['input_ids']

        batch_size = input_ids.shape[0]
        n_digit = self.tokenizer.n_digit
        user_ids = batch.get('user_ids')
        dialogue_sid_bias = self.build_dialogue_sid_bias(user_ids)
        dialogue_intent_sid_bias = (
            self.build_dialogue_multi_intent_sid_bias(user_ids))
        if user_ids is not None:
            user_ids = user_ids.repeat_interleave(num_beams, dim=0)
        history_item_ids = history_item_ids.repeat_interleave(
            num_beams, dim=0)
        if dialogue_sid_bias is not None:
            dialogue_sid_bias = dialogue_sid_bias.repeat_interleave(
                num_beams, dim=0)
        if dialogue_intent_sid_bias is not None:
            dialogue_intent_sid_bias = (
                dialogue_intent_sid_bias.repeat_interleave(
                    num_beams, dim=0))

        input_ids, attention_mask, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
            input_ids, attention_mask, batch_size, num_beams)
        his_len = input_ids.shape[1] - n_digit

        for pos in range(n_digit):
            outputs = self.llada_forward(input_ids=input_ids,
                                         attention_mask=attention_mask.long(),
                                         user_ids=user_ids,
                                         history_item_ids=history_item_ids)

            logits = outputs.logits
            logits = self.apply_dialogue_sid_bias(
                logits, dialogue_sid_bias)
            logits = self.apply_dialogue_multi_intent_sid_bias(
                logits, dialogue_intent_sid_bias)
            if self.user_condition_guidance_scale != 0.0:
                uncond_outputs = self.llada_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask.long(),
                    user_ids=user_ids,
                    history_item_ids=history_item_ids,
                    disable_user_conditions=True,
                )
                logits = uncond_outputs.logits + \
                    self.user_condition_guidance_scale * \
                    (logits - uncond_outputs.logits)
            logits = logits / self.temperature

            transfer_index = torch.full((input_ids.shape[0], ),
                                        his_len + pos,
                                        dtype=torch.long,
                                        device=input_ids.device)
            input_ids, beam_scores = self.beam_search_step_sid_trie(
                logits, input_ids, transfer_index, beam_scores,
                beam_idx_offset, batch_size, num_beams)

        selection_mask = torch.zeros(batch_size,
                                     num_beams,
                                     dtype=bool,
                                     device=input_ids.device)
        selection_mask[:, :num_return_sequences] = True

        label_ids = input_ids[:, -n_digit:]
        label_ids = label_ids[selection_mask.view(-1), :].reshape(
            -1, num_return_sequences, n_digit)

        if return_scores:
            selected_scores = beam_scores.reshape(batch_size, num_beams)[
                :, :num_return_sequences]
            return label_ids, selected_scores
        return label_ids

    def beam_search_step_sid_trie(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        transfer_index: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_idx_offset: torch.Tensor,
        batch_size,
        num_beams,
    ):
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits.gather(
            index=transfer_index.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, vocab_size),
            dim=1).squeeze(1)
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        his_len = input_ids.shape[1] - self.tokenizer.n_digit
        pos = int((transfer_index[0] - his_len).item())
        candidate_beams = []
        candidate_scores = []
        for row in range(batch_size):
            row_candidates = []
            for beam in range(num_beams):
                flat_idx = row * num_beams + beam
                base_score = beam_scores[flat_idx]
                if base_score.item() <= -1e8:
                    continue
                prefix = tuple(
                    int(x)
                    for x in input_ids[flat_idx, his_len:his_len + pos].tolist())
                allowed = self.sid_trie_children.get(prefix, [])
                if not allowed:
                    continue
                allowed_t = torch.tensor(allowed,
                                         dtype=torch.long,
                                         device=input_ids.device)
                scores = next_token_scores[flat_idx, allowed_t] + base_score
                k = min(num_beams, allowed_t.numel())
                top_scores, top_idx = scores.topk(k,
                                                  dim=0,
                                                  largest=True,
                                                  sorted=True)
                top_tokens = allowed_t[top_idx]
                for score, token in zip(top_scores, top_tokens):
                    row_candidates.append(
                        (float(score.item()), flat_idx, int(token.item())))

            row_candidates.sort(key=lambda item: item[0], reverse=True)
            row_candidates = row_candidates[:num_beams]
            if not row_candidates:
                for beam in range(num_beams):
                    flat_idx = row * num_beams + beam
                    row_candidates.append(
                        (float(beam_scores[flat_idx].item()), flat_idx,
                         int(self.mask_token_id)))

            row_input_ids = []
            row_scores = []
            for score, source_idx, token in row_candidates:
                new_ids = input_ids[source_idx].clone()
                new_ids[his_len + pos] = token
                row_input_ids.append(new_ids)
                row_scores.append(score)

            if len(row_input_ids) < num_beams:
                pad_num = num_beams - len(row_input_ids)
                row_input_ids.extend([row_input_ids[-1].clone()] * pad_num)
                row_scores.extend([-1e9] * pad_num)

            candidate_beams.extend(row_input_ids[:num_beams])
            candidate_scores.extend(row_scores[:num_beams])

        input_ids = torch.stack(candidate_beams, dim=0)
        beam_scores = torch.tensor(candidate_scores,
                                   dtype=logits.dtype,
                                   device=logits.device)

        return input_ids, beam_scores

    def beam_search_step(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        transfer_index: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_idx_offset: torch.Tensor,
        batch_size,
        num_beams,
        user_ids: Optional[torch.Tensor] = None,
    ):
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits.gather(
            index=transfer_index.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, vocab_size),
            dim=1).squeeze(1)

        # Calculate log softmax over the last dimension
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        his_len = input_ids.shape[1] - self.tokenizer.n_digit
        valid_tokens = self.posValidTokens[transfer_index - his_len]

        valid_tokens = valid_tokens.masked_fill(valid_tokens == 0,
                                                self.mask_token_id)
        next_token_scores = next_token_scores.gather(index=valid_tokens, dim=1)
        if (self.use_qwen_teacher_beam_prior
                and self.qwen_beam_prior_scale != 0.0
                and user_ids is not None):
            next_token_scores = next_token_scores + \
                self.qwen_teacher_beam_boost(valid_tokens, transfer_index,
                                             his_len, user_ids)
        next_token_scores = next_token_scores.masked_fill(
            valid_tokens == self.mask_token_id, -np.inf)
        vocab_size = next_token_scores.shape[-1]

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
            next_token_scores)
        next_token_scores = next_token_scores.view(batch_size,
                                                   num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores,
                                                    2 * num_beams,
                                                    dim=1,
                                                    largest=True,
                                                    sorted=True)

        next_indices = torch.div(next_tokens,
                                 vocab_size,
                                 rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        input_ids = input_ids[beam_idx + beam_idx_offset]
        transfer_index = transfer_index[beam_idx + beam_idx_offset]

        valid_tokens = self.posValidTokens[transfer_index - his_len]
        beam_next_tokens = valid_tokens.gather(
            index=beam_next_tokens.unsqueeze(1), dim=1).squeeze(1)

        x_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        input_ids[x_index, transfer_index] = beam_next_tokens

        return input_ids, beam_scores

    def qwen_teacher_beam_boost(
        self,
        valid_tokens: torch.Tensor,
        transfer_index: torch.Tensor,
        his_len: int,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_ids = user_ids.long().clamp(
            min=0, max=self.qwen_teacher_item_ids.shape[0] - 1)
        teacher_items = self.qwen_teacher_item_ids[user_ids]
        teacher_scores = self.qwen_teacher_scores[user_ids]
        valid_items = teacher_items > 0
        covered = valid_items.any(dim=1)
        if not covered.any():
            return torch.zeros_like(valid_tokens, dtype=torch.float)

        weights = teacher_scores.masked_fill(~valid_items, -1e4)
        weights = F.softmax(weights, dim=-1)
        weights = weights * covered.unsqueeze(1)

        teacher_items = teacher_items.clamp(min=0,
                                            max=self.item_id2tokens.shape[0] -
                                            1)
        teacher_tokens = self.item_id2tokens[teacher_items]
        token_pos = (transfer_index - his_len).long().clamp(
            min=0, max=self.tokenizer.n_digit - 1)
        token_pos = token_pos[:, None, None].expand(-1,
                                                    teacher_tokens.shape[1],
                                                    1)
        pos_tokens = teacher_tokens.gather(dim=2, index=token_pos).squeeze(-1)

        prior = torch.zeros((valid_tokens.shape[0], self.tokenizer.vocab_size),
                            dtype=weights.dtype,
                            device=weights.device)
        prior.scatter_add_(1, pos_tokens, weights)
        gathered = prior.gather(1, valid_tokens.clamp(min=0))
        boost = torch.log1p(self.qwen_beam_prior_alpha * gathered)
        return self.qwen_beam_prior_scale * boost

    def prepare_beam_search_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size,
        num_beams,
    ):
        beam_label_ids = torch.ones(
            (batch_size * num_beams, self.tokenizer.n_digit),
            device=self.config['device'],
            dtype=torch.long) * self.mask_token_id
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        all_input_ids = torch.cat([input_ids, beam_label_ids], dim=-1)

        beam_label_attention_mask = torch.ones_like(
            beam_label_ids, device=self.config['device'], dtype=torch.long)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        all_attention_mask = torch.cat(
            [attention_mask, beam_label_attention_mask], dim=-1)

        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view((batch_size * num_beams, ))

        beam_idx_offset = torch.arange(
            batch_size, device=self.config['device']).repeat_interleave(
                num_beams) * num_beams

        return all_input_ids, all_attention_mask, initial_beam_scores, beam_idx_offset

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
