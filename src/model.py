import math
import torch
from torch import nn
from typing import Optional, Tuple
from transformers import AutoConfig

from sentence_transformers import SentenceTransformer

class BertAttention(nn.Module):
    def __init__(self, original_self_attention, temperature=1.0):
        super().__init__()
        self.original_self_attention = original_self_attention
        self.query = original_self_attention.query
        self.key = original_self_attention.key
        self.value = original_self_attention.value
        self.dropout = original_self_attention.dropout
        self.attention_head_size = original_self_attention.attention_head_size
        self.num_attention_heads = original_self_attention.num_attention_heads
        self.transpose_for_scores = original_self_attention.transpose_for_scores
        self.is_decoder = original_self_attention.is_decoder
        
        self.temperature = temperature

    def forward( self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.original_self_attention.position_embedding_type == "relative_key" or self.original_self_attention.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.original_self_attention.distance_embedding(distance + self.original_self_attention.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.original_self_attention.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.original_self_attention.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores/self.temperature, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.original_self_attention.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    

class T5Attention(nn.Module):
    def __init__(self, original_self_attention, temperature=1.0):
        super().__init__()
        self.original_self_attention = original_self_attention
        self.temperature = temperature

    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
        ):
            """
            Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
            """
            # Input is (batch_size, seq_length, dim)
            # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
            # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
            batch_size, seq_length = hidden_states.shape[:2]

            real_seq_length = seq_length

            if past_key_value is not None:
                if len(past_key_value) != 2:
                    raise ValueError(
                        f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                    )
                real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

            key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

            def shape(states):
                """projection"""
                return states.view(batch_size, -1, self.original_self_attention.n_heads, self.original_self_attention.key_value_proj_dim).transpose(1, 2)

            def unshape(states):
                """reshape"""
                return states.transpose(1, 2).contiguous().view(batch_size, -1, self.original_self_attention.inner_dim)

            def project(hidden_states, proj_layer, key_value_states, past_key_value):
                """projects hidden states correctly to key/query states"""
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(hidden_states))
                elif past_key_value is None:
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))

                if past_key_value is not None:
                    if key_value_states is None:
                        # self-attn
                        # (batch_size, n_heads, key_length, dim_per_head)
                        hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                    elif past_key_value.shape[2] != key_value_states.shape[1]:
                        # checking that the `sequence_length` of the `past_key_value` is the same as
                        # the provided `key_value_states` to support prefix tuning
                        # cross-attn
                        # (batch_size, n_heads, seq_length, dim_per_head)
                        hidden_states = shape(proj_layer(key_value_states))
                    else:
                        # cross-attn
                        hidden_states = past_key_value
                return hidden_states

            # get query states
            query_states = shape(self.original_self_attention.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

            # get key/value states
            key_states = project(
                hidden_states, self.original_self_attention.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )
            value_states = project(
                hidden_states, self.original_self_attention.v, key_value_states, past_key_value[1] if past_key_value is not None else None
            )

            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            if position_bias is None:
                if not self.original_self_attention.has_relative_attention_bias:
                    position_bias = torch.zeros(
                        (1, self.original_self_attention.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                    )
                    if self.original_self_attention.gradient_checkpointing and self.training:
                        position_bias.requires_grad = True
                else:
                    position_bias = self.original_self_attention.compute_bias(real_seq_length, key_length, device=scores.device)

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

                if mask is not None:
                    position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            if self.original_self_attention.pruned_heads:
                mask = torch.ones(position_bias.shape[1])
                mask[list(self.original_self_attention.pruned_heads)] = 0
                position_bias_masked = position_bias[:, mask.bool()]
            else:
                position_bias_masked = position_bias

            scores += position_bias_masked
            attn_weights = nn.functional.softmax(scores.float()/self.temperature, dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.original_self_attention.dropout, training=self.training
            )  # (batch_size, n_heads, seq_length, key_length)

            if layer_head_mask is not None:
                attn_weights = attn_weights * layer_head_mask

            attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
            attn_output = self.original_self_attention.o(attn_output)

            present_key_value_state = (key_states, value_states) if (self.original_self_attention.is_decoder and use_cache) else None
            outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

            if output_attentions:
                outputs = outputs + (attn_weights,)
            return outputs



class TemperatureSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path, temperature=1.0):
        super().__init__(model_name_or_path, trust_remote_code=True)
        if "gte" in model_name_or_path:
            self.model_type = "gte"
        else:
            self.model_type = AutoConfig.from_pretrained(model_name_or_path).model_type
        self.temperature = temperature
        model = self[0].auto_model

        if self.model_type in ['bert', 'roberta']:
            for name, module in model.named_modules():
                if name.endswith('attention.self') is True:
                    parent_name, attribute_name  = name.rsplit('.', 1)
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    setattr(parent_module, attribute_name, BertAttention(module, temperature=self.temperature))
        elif self.model_type in ['t5']:
            for name, module in model.named_modules():
                if name.endswith('SelfAttention') is True:
                    parent_name, attribute_name  = name.rsplit('.', 1)
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    setattr(parent_module, attribute_name, T5Attention(module, temperature=self.temperature))
            
