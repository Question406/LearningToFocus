import math
import os
import warnings
import random
from typing import Optional, Union, Tuple, List, Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import transformers
from transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallModel, \
    PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, \
    Seq2SeqModelOutput
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotLearnedPositionalEmbedding, \
    BlenderbotEncoderLayer, _expand_mask, _make_causal_mask, shift_tokens_right


class copyBlenderbotConfig(PretrainedConfig):
    model_type = "copyblenderbotconfig"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # default is based on facebook/blenderbot-400M-distill
    def __init__(
            self,
            vocab_size=8011,
            max_position_embeddings=256,
            encoder_layers=2,
            encoder_ffn_dim=5120,
            encoder_attention_heads=32,
            decoder_layers=12,
            decoder_ffn_dim=5120,
            decoder_attention_heads=32,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=1280,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            decoder_start_token_id=1,
            classifier_dropout=0.0,
            scale_embedding=True,
            gradient_checkpointing=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            encoder_no_repeat_ngram_size=3,
            forced_eos_token_id=2,
            attn_type='none',
            attn_init_type='none',
            **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        
        self.attn_type = attn_type
        self.attn_init_type = 'none' if attn_type == 'none' else attn_init_type
        assert self.attn_type in {
            'none', 'input', 'layer', 'attention', 'mul_input', 'mul_layer', 'lin_input', 'lin_layer', 'mul_flayer'
        }


class copyBlenderbotPreTrainedModel(transformers.modeling_utils.PreTrainedModel):
    config_class = copyBlenderbotConfig
    base_model_prefix = "model"
    
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
    
    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs


# adapted from modeling_utils.py:_get_resized_embedding
def _get_resized_position_embedding(device, old_embeddings: BlenderbotLearnedPositionalEmbedding,
                                    new_num_tokens: int) -> BlenderbotLearnedPositionalEmbedding:
    if new_num_tokens is None:
        return old_embeddings
    
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    if old_num_tokens == new_num_tokens:
        return old_embeddings
    
    if not isinstance(old_embeddings, BlenderbotLearnedPositionalEmbedding):
        raise TypeError(
            f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {BlenderbotLearnedPositionalEmbedding}. "
            f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {BlenderbotLearnedPositionalEmbedding}. "
        )
    
    # Build new embeddings
    new_embeddings = BlenderbotLearnedPositionalEmbedding(new_num_tokens, old_embedding_dim).to(device)
    
    # Copy token embeddings from the previous weights
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
    
    return new_embeddings


def _get_attn_embedding(attn_init_type, vec_dim):
    if attn_init_type == 'none':
        return None
    if attn_init_type == 'default' or attn_init_type == 'normal':
        return nn.Embedding(2, vec_dim)
    elif attn_init_type == 'constant_1':
        newweight = torch.ones(2, vec_dim, dtype=torch.float)
        newembed = nn.Embedding(2, vec_dim, _weight=newweight)
        return newembed
    elif attn_init_type == 'xavier':
        newweight = torch.empty(2, vec_dim)
        # torch.nn.init.xavier_uniform_(newweight)
        newweight.normal_(mean=0.0, std=0.02)
        newembed = nn.Embedding(2, vec_dim, _weight=newweight)
        return newembed
    elif attn_init_type == 'empty':
        newweight = torch.zeros(2, vec_dim)
        newembed = nn.Embedding(2, vec_dim, _weight=newweight)
        return newembed
    else:
        raise ValueError("Unknown attn_init_type {}".format(attn_init_type))


# ATTN_CONST = 3.3610526315789477
ATTN_CONST = 3.0326394770521578


def setATTN_CONST(newval):
    global ATTN_CONST
    ATTN_CONST = newval


def ATTN_FUNCTION(attn_input_ids):
    # CONST = 2999.
    return ATTN_CONST * attn_input_ids


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Blenderbot
class copyBlenderbotAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_input_ids: Optional[torch.Tensor] = None,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if attn_input_ids is not None:
            attn_weights = attn_weights + ATTN_FUNCTION(
                attn_input_ids)  # only for batch_size 1, TODO: to be random batch_size
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.bmm(attn_probs, value_states)
        
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Blenderbot


class copyBlenderbotDecoderLayer(nn.Module):
    def __init__(self, config: copyBlenderbotConfig):
        super().__init__()
        self.attn_type = config.attn_type
        self.embed_dim = config.d_model
        
        self.self_attn = copyBlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = copyBlenderbotAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            encoder_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attn_input_ids=attn_input_ids if self.attn_type == 'attention' else None,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            
            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class copyBlenderbotEncoderLayer(nn.Module):
    def __init__(self, config: copyBlenderbotConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = copyBlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    # def add_encoder_layer_bias(self):
    #     self.attn_embed = nn.Embedding(2, self.embed_dim)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs


class copyBlenderbotEncoder(copyBlenderbotPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BlenderbotEncoderLayer`.

    Args:
        config: BlenderbotConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """
    
    def __init__(self, config: copyBlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        
        embed_dim = config.d_model
        self.embed_dim = embed_dim
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        
        self.input_embeds_grads = None
        
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        
        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        
        self.attn_embed = None
        self.layer_attn_embeds = None
        
        self.attn_type = config.attn_type
        if config.attn_type == 'input':
            self.attn_embed = _get_attn_embedding(config.attn_init_type, config.d_model).to(self.device)
        elif config.attn_type == 'layer':
            assert config.attn_init_type == 'xavier'  # we can't get a good result with normal initialize
            self.layer_attn_embeds = nn.ModuleList([])
            for _ in range(config.encoder_layers):
                newembed = _get_attn_embedding(config.attn_init_type, config.d_model).to(self.device)
                self.layer_attn_embeds.append(newembed)
        elif config.attn_type == 'mul_input':
            bias_embed = _get_attn_embedding(config.attn_init_type, config.d_model).to(self.device)
            mul_embed = _get_attn_embedding('constant_1', config.d_model).to(self.device)
            self.attn_embed = nn.ModuleList([mul_embed, bias_embed])
        elif config.attn_type == 'mul_layer':
            assert config.attn_init_type == 'xavier'  # we can't get a good result with normal initialize
            self.layer_attn_embeds = nn.ModuleList([])
            for _ in range(config.encoder_layers):
                bias_embed = _get_attn_embedding(config.attn_init_type, config.d_model).to(self.device)
                mul_embed = _get_attn_embedding('constant_1', config.d_model).to(self.device)
                self.layer_attn_embeds.append(nn.ModuleList([mul_embed, bias_embed]))
        elif config.attn_type == 'mul_flayer':
            assert config.attn_init_type == 'xavier'  # we can't get a good result with normal initialize
            self.layer_attn_embeds = nn.ModuleList([])
            for _ in range(1):  # only the last encoder layer
                bias_embed = _get_attn_embedding(config.attn_init_type, config.d_model).to(self.device)
                mul_embed = _get_attn_embedding('constant_1', config.d_model).to(self.device)
                self.layer_attn_embeds.append(nn.ModuleList([mul_embed, bias_embed]))
        elif config.attn_type == 'lin_input':
            assert config.attn_init_type == 'xavier'  # we can't get a good result with normal initialize
            attn_lin = nn.Linear(config.d_model, config.d_model).to(self.device)
            attn_embed = _get_attn_embedding(config.attn_init_type, config.d_model).to(self.device)
            norm = nn.LayerNorm(config.d_model)
            self.attn_embed = nn.ModuleList([attn_embed, attn_lin, norm])
        elif config.attn_type == 'lin_layer':
            assert config.attn_init_type == 'xavier'  # we can't get a good result with normal initialize
            self.layer_attn_embeds = nn.ModuleList([])
            for _ in range(config.encoder_layers):
                attn_lin = nn.Linear(config.d_model, config.d_model).to(self.device)
                attn_embed = _get_attn_embedding(config.attn_init_type, config.d_model).to(self.device)
                norm = nn.LayerNorm(config.d_model)
                self.layer_attn_embeds.append(nn.ModuleList([attn_embed, attn_lin, norm]))
        
        self.layers = nn.ModuleList([copyBlenderbotEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.init_weights()
    
    def clear_input_embeds_grad(self):
        self.input_embeds_grads = None
    
    def resize_position_embedding(self, new_num_tokens: int):
        self.embed_positions = _get_resized_position_embedding(self.device, self.embed_positions, new_num_tokens)
        self.max_source_positions = new_num_tokens
    
    def get_layer_attn_embeds(self):
        return self.layer_attn_embeds
    
    def get_attn_embed(self):
        if self.attn_type == 'input' or self.attn_type == 'mul_input' or self.attn_type == 'lin_input':
            return self.attn_embed
        elif self.attn_type == 'layer' or self.attn_type == 'mul_layer' or self.attn_type == 'lin_layer' or self.attn_type == 'mul_flayer':
            return self.layer_attn_embeds
        else:
            return None
        # assert False, 'Should not be here'
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            attn_input_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        encoder_states = () if output_hidden_states else None
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        if output_hidden_states:
            def store_grad(raw_grad):
                self.input_embeds_grads = raw_grad.clone()
            
            inputs_embeds.register_hook(lambda grad: store_grad(grad))
            encoder_states = encoder_states + (inputs_embeds,)
        
        embed_pos = self.embed_positions(input_shape)
        
        hidden_states = inputs_embeds + embed_pos
        
        if attn_input_ids is not None:
            if self.attn_embed is not None:
                if self.attn_type == 'input':
                    attn_embeds = self.attn_embed(
                        attn_input_ids) * self.embed_scale  # keep attn_embedding similar to word embedding
                    hidden_states = hidden_states + attn_embeds
                elif self.attn_type == 'mul_input':
                    hidden_states = self.attn_embed[0](attn_input_ids) * hidden_states + self.attn_embed[1](
                        attn_input_ids) * self.embed_scale
                elif self.attn_type == 'lin_input':
                    attn_hidden_states = self.attn_embed[1](self.attn_embed[0](attn_input_ids)) * self.embed_scale
                    attn_hidden_states = self.attn_embed[2](attn_hidden_states)
                    hidden_states = hidden_states + attn_hidden_states
        
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        
        all_attentions = () if output_attentions else None
        
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        
        for idx, encoder_layer in enumerate(self.layers):
            if self.layer_attn_embeds is not None and attn_input_ids is not None:
                if self.attn_type == 'layer':
                    attn_embeds = self.layer_attn_embeds[idx](
                        attn_input_ids) * self.embed_scale  # keep attn_embedding similar to word embedding
                    hidden_states = hidden_states + attn_embeds
                elif self.attn_type == 'mul_layer':
                    # attn_embeds = self.layer_attn_embeds[idx](attn_input_ids) * hidden_states * self.embed_scale
                    hidden_states = self.layer_attn_embeds[idx][0](attn_input_ids) * hidden_states + \
                                    self.layer_attn_embeds[idx][1](attn_input_ids) * self.embed_scale
                elif self.attn_type == 'mul_flayer':
                    assert len(self.layer_attn_embeds) == 1
                    if idx == len(self.layers) - 1:
                        hidden_states = self.layer_attn_embeds[0][0](attn_input_ids) * hidden_states + \
                                        self.layer_attn_embeds[0][1](attn_input_ids) * self.embed_scale
                elif self.attn_type == 'lin_layer':
                    attn_hidden_states = self.layer_attn_embeds[idx][1](
                        self.layer_attn_embeds[idx][0](attn_input_ids)) * self.embed_scale
                    attn_hidden_states = self.layer_attn_embeds[idx][2](attn_hidden_states)
                    hidden_states = hidden_states + attn_hidden_states
                else:
                    assert False, "Should be specific attn type"
            
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:
                    
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)
                        
                        return custom_forward
                    
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )
                
                hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class copyBlenderbotDecoder(copyBlenderbotPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BlenderbotDecoderLayer`

    Args:
        config: BlenderbotConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """
    
    def __init__(self, config: copyBlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        
        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([copyBlenderbotDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def resize_position_embedding(self, new_num_tokens: int):
        self.embed_positions = _get_resized_position_embedding(self.device, self.embed_positions, new_num_tokens)
        self.max_target_positions = new_num_tokens
    
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)
        
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        
        return combined_attention_mask
    
    def forward(
            self,
            input_ids=None,
            attn_input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        
        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)
        
        hidden_states = inputs_embeds + positions
        
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None
        
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                
                if use_cache:
                    use_cache = False
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)
                    
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    attn_input_ids=attn_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        
        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class copyBlenderbotModel(copyBlenderbotPreTrainedModel):
    
    def __init__(self, config: copyBlenderbotConfig):
        super().__init__(config)
        
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        
        self.encoder = copyBlenderbotEncoder(config, self.shared)
        self.decoder = copyBlenderbotDecoder(config, self.shared)
        
        self.init_weights()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical checkpoint `facebook/small_blenderbot-90M` with `BlenderbotSmallModel.from_pretrained('facebook/small_blenderbot-90M')` instead.",
                FutureWarning,
            )
            return BlenderbotSmallModel.from_pretrained(pretrained_model_name_or_path)
        
        return super(copyBlenderbotModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
    def get_attn_embed(self):
        return self.encoder.get_attn_embed()
    
    def set_layer_attn_embeds(self, add=False):
        self.encoder.set_layer_attn_embeds(add)
    
    def get_layer_attn_embeds(self):
        return self.encoder.get_layer_attn_embeds()
    
    def get_input_embeddings(self):
        return self.shared
    
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            attn_input_ids=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                attn_input_ids=attn_input_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attn_input_ids=attn_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            res = decoder_outputs + encoder_outputs
        else:
            res = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        
        return res


class copyBlenderbotForConditionalGeneration(copyBlenderbotPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]
    
    def __init__(self, config: copyBlenderbotConfig):
        super().__init__(config)
        self.model = copyBlenderbotModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        self.init_weights()
    
    def get_attn_embed(self):
        return self.model.get_attn_embed()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical checkpoint `facebook/small_blenderbot-90M` with `BlenderbotSmallForConditionalGeneration.from_pretrained('facebook/small_blenderbot-90M')` instead.",
                FutureWarning,
            )
            return BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
        
        return super(copyBlenderbotForConditionalGeneration, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
    
    def get_encoder(self):
        return self.model.get_encoder()
    
    def get_decoder(self):
        return self.model.get_decoder()
    
    def add_encoder_layer_bias(self):
        self.get_encoder().set_layer_attn_embeds()
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings
    
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def resize_position_embedding(self, new_num_tokens: int) -> nn.Embedding:
        self.get_encoder().resize_position_embedding(new_num_tokens)
        self.get_decoder().resize_position_embedding(new_num_tokens)
        self.config.max_position_embeddings = new_num_tokens
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            attn_input_ids=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        # if input_ids is not None:
        #     print('input_ids', input_ids)
        # if decoder_input_ids is not None:
        #     print('decoder_input_ids', decoder_input_ids)
        # if labels is not None:
        #     print('labels', labels)
        # if attn_input_ids is not None:
        #     print('attn_input_ids', attn_input_ids)
        # x = input('--blenderbot forward--')
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            attn_input_ids=attn_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        res = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            'attn_input_ids': kwargs.get('attn_input_ids', None),
        }
        
        # print(res)
        # x = input("--WTF IS GOING ON--")
        
        return res
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
