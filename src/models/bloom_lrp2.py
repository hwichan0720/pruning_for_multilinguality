import warnings
from typing import Dict, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import BloomForCausalLM, BloomModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LRP2BloomModel(BloomModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tgt_lang_vectors: Optional[Dict[int, torch.Tensor]] = None,
        src_lang_vectors: Optional[Dict[int, torch.Tensor]] = None,
        lirp_layer: Optional[int] = None,
        lsrp_layer: Optional[int] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if lirp_layer:
            assert (
                lirp_layer < lsrp_layer
            ), f"Please set lirp_layer ({lirp_layer}), lsrp_layer ({lsrp_layer}) to satisfy lirp_layer < lsrp_layer"
            assert (
                src_lang_vectors[lirp_layer].shape == tgt_lang_vectors[lirp_layer].shape
            ), "Not match the shapes"

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), device=hidden_states.device
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(
            attention_mask, self.num_heads, dtype=hidden_states.dtype
        )

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # add or subtractlanguage vector
            if lirp_layer and i == lirp_layer:
                assert (
                    hidden_states.shape == tgt_lang_vectors[lirp_layer].shape
                ), "Not match the shapes"
                hidden_states -= tgt_lang_vectors[lirp_layer]
                hidden_states += src_lang_vectors[lirp_layer]
            if lsrp_layer and i == lsrp_layer:
                assert (
                    hidden_states.shape == tgt_lang_vectors[lsrp_layer].shape
                ), "Not match the shapes"
                hidden_states -= src_lang_vectors[lsrp_layer]
                hidden_states += tgt_lang_vectors[lsrp_layer]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        if lsrp_layer and i + 1 == lsrp_layer:
            assert (
                hidden_states.shape == tgt_lang_vectors[lsrp_layer].shape
            ), "Not match the shapes"
            hidden_states -= src_lang_vectors[lsrp_layer]
            hidden_states += tgt_lang_vectors[lsrp_layer]

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LRP2BloomForCausalLM(BloomForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LRP2BloomModel(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
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
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length),
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


if __name__ == "__main__":
    import numpy as np
    from transformers import AutoTokenizer

    model = LRP2BloomForCausalLM.from_pretrained("bigscience/bloom-3B")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3B")

    en_vector = np.load(
        "/home/hwichan/prune_for_multilingual/outputs/bigscience/bloom-3B/lang_vector/en_vector.npy"
    )
    zh_vector = np.load(
        "/home/hwichan/prune_for_multilingual/outputs/bigscience/bloom-3B/lang_vector/zh_vector.npy"
    )
    en_vector = torch.from_numpy(en_vector)
    zh_vector = torch.from_numpy(zh_vector)

    inputs = ["zh zh en en zh", "zh en en zh"]
    # tokenized_inputs = ['zh zh en en zh', '<pad>zh en en zh']
    langs = torch.tensor([[1, 1, 0, 0, 1], [0, 1, 0, 0, 1]])
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
