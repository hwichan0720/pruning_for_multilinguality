import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import XGLMForCausalLM, XGLMModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class LRP2XGLMModel(XGLMModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tgt_lang_vectors: Optional[Dict[int, torch.Tensor]] = None,
        src_lang_vectors: Optional[Dict[int, torch.Tensor]] = None,
        lirp_layer: Optional[int] = None,
        lsrp_layer: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        if lirp_layer:
            assert (
                lirp_layer < lsrp_layer
            ), f"Please set lirp_layer ({lirp_layer}), lsrp_layer ({lsrp_layer}) to satisfy lirp_layer < lsrp_layer"
            assert (
                src_lang_vectors[lirp_layer].shape == tgt_lang_vectors[lirp_layer].shape
            ), "Not match the shapes"

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                input_shape[-1] + past_key_values_length,
                dtype=torch.long,
                device=(
                    input_ids.device if input_ids is not None else inputs_embeds.device
                ),
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        else:
            position_ids = position_ids.view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        hidden_states = inputs_embeds + self.embed_positions(
            position_ids, past_key_values_length
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=float(self.dropout), training=self.training
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing`. Setting `use_cache ="
                    " False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add or subtractlanguage vector
            if lirp_layer and idx == lirp_layer:
                assert (
                    hidden_states.shape == tgt_lang_vectors[lirp_layer].shape
                ), "Not match the shapes"
                hidden_states -= tgt_lang_vectors[lirp_layer]
                hidden_states += src_lang_vectors[lirp_layer]
            if lsrp_layer and idx == lsrp_layer:
                assert (
                    hidden_states.shape == tgt_lang_vectors[lsrp_layer].shape
                ), "Not match the shapes"
                hidden_states -= src_lang_vectors[lsrp_layer]
                hidden_states += tgt_lang_vectors[lsrp_layer]

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

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
                    (
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
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

        hidden_states = self.layer_norm(hidden_states)

        if lsrp_layer and idx + 1 == lsrp_layer:
            assert (
                hidden_states.shape == tgt_lang_vectors[lsrp_layer].shape
            ), "Not match the shapes"
            hidden_states -= src_lang_vectors[lsrp_layer]
            hidden_states += tgt_lang_vectors[lsrp_layer]

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class LRP2XGLMForCausalLM(XGLMForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LRP2XGLMModel(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tgt_lang_vectors: Optional[torch.Tensor] = None,
        src_lang_vectors: Optional[torch.Tensor] = None,
        lirp_layer: Optional[int] = None,
        lsrp_layer: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            tgt_lang_vectors=tgt_lang_vectors,
            src_lang_vectors=src_lang_vectors,
            lirp_layer=lirp_layer,
            lsrp_layer=lsrp_layer,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # shift labels and add a pad token to the end
            shift_labels = labels.new_zeros(labels.shape)
            shift_labels[:, :-1] = labels[:, 1:].clone()
            shift_labels[:, -1] = self.config.pad_token_id

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


if __name__ == "__main__":
    import numpy as np
    from transformers import AutoTokenizer

    model = LRP2XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-2.9B")

    en_vector = np.load(
        "/home/hwichan/prune_for_multilingual/outputs/facebook/xglm-2.9B/lang_vector/en_vector.npy"
    )
    zh_vector = np.load(
        "/home/hwichan/prune_for_multilingual/outputs/facebook/xglm-2.9B/lang_vector/zh_vector.npy"
    )
    en_vector = torch.from_numpy(en_vector)
    zh_vector = torch.from_numpy(zh_vector)

    inputs = ["zh zh en en zh", "zh en en zh"]
    langs = torch.tensor([[0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0]])
    lirp_layer = 3
    lsrp_layer = 5

    tgt_lang_vectors = {}
    src_lang_vectors = {}
    for layer in [lirp_layer, lsrp_layer]:
        tgt_lang_vector = torch.zeros(len(langs), len(langs[0]), zh_vector.shape[1]).to(
            zh_vector.dtype
        )
        tgt_lang_vector[torch.where(langs == 1)] = zh_vector[layer]
        src_lang_vector = torch.zeros(len(langs), len(langs[0]), en_vector.shape[1]).to(
            en_vector.dtype
        )
        src_lang_vector[torch.where(langs == 1)] = en_vector[layer]

        if len(tgt_lang_vector.shape) < 3:
            tgt_lang_vector = tgt_lang_vector.unsqueeze(0)
            src_lang_vector = src_lang_vector.unsqueeze(0)
        tgt_lang_vectors[layer] = tgt_lang_vector
        src_lang_vectors[layer] = src_lang_vector

    tokenized_inputs = tokenizer(inputs, padding=True, return_tensors="pt")
    outs = model(
        input_ids=tokenized_inputs["input_ids"],
        attention_mask=tokenized_inputs["attention_mask"],
        tgt_lang_vectors=tgt_lang_vectors,
        src_lang_vectors=src_lang_vectors,
        lirp_layer=lirp_layer,
        lsrp_layer=lsrp_layer,
    )
