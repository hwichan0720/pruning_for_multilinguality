from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

verbalizer = {
    "en": {"entailment": "Yes", "contradiction": "No", "neutral": "Also"},
    "fr": {"entailment": "Oui", "contradiction": "Non", "neutral": "De plus"},
    "es": {"entailment": "Sí", "contradiction": "No", "neutral": "Además"},
    "zh": {
        "entailment": "由此可知",
        "contradiction": "所以，不可能",
        "neutral": "同时，",
    },
    "tr": {"entailment": "Evet", "contradiction": "Hayır", "neutral": "Ayrıca"},
}
templates = {
    "en": "right?",
    "es": "¿verdad?",
    "zh": "",
    "fr": "non?",
    "tr": "değil mi?",
}


def create_xnil_context(
    example: Dict, label_names: List[str], template_lang: str = "en"
):
    lang_template = templates[template_lang]
    label_name = label_names[example["label"]]
    target_word = verbalizer[template_lang][label_name]
    context = (
        example["premise"]
        + f" , {lang_template} {target_word} , "
        + example["hypothesis"]
    )
    return context


def create_xnil_v2_context(
    example: Dict, label_names: List[str], template_lang: str = "en"
):
    label_name = label_names[example["label"]]
    target_word = verbalizer[template_lang][label_name]
    context = example["premise"] + example["hypothesis"] + " " + target_word
    return context


def create_marc2_context(
    example: Dict, label_names: List[str], template_lang: str = "en"
):
    target_word = label_names[example["label"]]
    context = " ".join(example["sentence"].split()[:128]) + " It is " + target_word
    return context


def create_xnil_prompt_per_label(
    examples,
    contexts: List[str] = [],
    template_lang: str = "en",
    do_mask: bool = False,
    tokenizer=None,
):
    lang_verbalizer = verbalizer[template_lang]
    for label in lang_verbalizer.keys():
        prompt = create_xnil_prompt(
            examples,
            label=label,
            contexts=contexts,
            template_lang=template_lang,
            do_mask=do_mask,
            tokenizer=tokenizer,
        )

        examples[label] = prompt["inputs"]
        examples[f"{label}_pre_range"] = prompt["pre_range"]
        examples[f"{label}_hyp_range"] = prompt["hyp_range"]

        if do_mask:
            examples[f"{label}_target_token_idx"] = prompt["target_token_idx"]

    return examples



def create_xnil_v2_prompt_per_label(
    examples, contexts: List[str] = [], template_lang: str = "en"
):
    lang_verbalizer = verbalizer[template_lang]
    for label in lang_verbalizer.keys():
        prompt = create_xnil_v2_prompt(
            examples, label=label, contexts=contexts, template_lang=template_lang
        )
        examples[label] = prompt
    return examples


def create_marc2_prompt_per_label(
    examples,
    contexts: List[str] = [],
    template_lang: str = "en",
    do_mask: bool = False,
    tokenizer=None,
):
    for label in ["negative", "positive"]:
        prompt = create_marc2_prompt(
            examples,
            label=label,
            contexts=contexts,
            template_lang=template_lang,
            do_mask=do_mask,
            tokenizer=tokenizer,
        )

        examples[label] = prompt["inputs"]
        examples[f"{label}_hyp_range"] = prompt["hyp_range"]
        examples[f"{label}_pre_range"] = prompt["pre_range"]
        if do_mask:
            examples[f"{label}_target_token_idx"] = prompt["target_token_idx"]
    return examples


def create_xnil_prompt(
    examples,
    label: str,
    contexts: List[str] = [],
    template_lang: str = "en",
    do_mask: bool = False,
    tokenizer=None,
):
    lang_template = templates[template_lang]
    lang_verbalizer = verbalizer[template_lang]
    word = lang_verbalizer[label]
    pre = examples["premise"]
    num_tokens_before_template = len(pre.split())
    num_template_tokens = len(f" , {lang_template} {word} , ".split())
    hyp = examples["hypothesis"]
    prompt1 = contexts + [pre + f" , {lang_template}"]
    prompt = prompt1 + [f" {word} , " + hyp]
    # prompt = contexts + [examples["premise"] + f", {lang_template} {word}, " + examples["hypothesis"]]
    tokenized_prompt = tokenizer(" ".join(prompt))
    target_token_idx = tokenized_prompt.word_ids(0)[-2] + 1
    pre_final_index = tokenized_prompt.word_ids(0).index(num_tokens_before_template)
    if tokenizer.padding_side == "right":
        pre_range = (1, pre_final_index)
    elif tokenizer.padding_side == "left":
        pre_range = (0, pre_final_index)

    hyp_first_index = tokenized_prompt.word_ids(0).index(
        num_tokens_before_template + num_template_tokens
    )
    hyp_range = (hyp_first_index, len(tokenized_prompt["input_ids"]))

    if not do_mask:
        return {
            "inputs": " ".join(prompt),
            "pre_range": pre_range,
            "hyp_range": hyp_range,
        }

    return {
        "inputs": " ".join(prompt),
        "target_token_idx": target_token_idx,
        "pre_range": pre_range,
        "hyp_range": hyp_range,
    }


def create_xnil_v2_prompt(
    examples, label: str, contexts: List[str] = [], template_lang: str = "en"
):
    lang_verbalizer = verbalizer[template_lang]
    word = lang_verbalizer[label]
    prompt = contexts + [f"{examples['premise']}  {examples['hypothesis']} {word}"]
    return "\n".join(prompt)


def create_marc2_prompt(
    examples,
    label: str,
    contexts: List[str] = [],
    template_lang: str = "en",
    do_mask: bool = False,
    tokenizer=None,
):
    sent = " ".join(examples["sentence"].split()[:128])
    assert len(sent.split()) <= 128, f"{len(sent.split())} {sent}"

    num_tokens_before_template = len(sent.split())
    prompt = contexts + [sent + " It is " + label]
    tokenized_prompt = tokenizer(" ".join(prompt))
    target_token_idx = tokenized_prompt.word_ids(0)[-2] + 1
    pre_final_index = tokenized_prompt.word_ids(0).index(num_tokens_before_template)
    if tokenizer.padding_side == "right":
        pre_range = (1, pre_final_index)
    elif tokenizer.padding_side == "left":
        pre_range = (0, pre_final_index)

    hyp_range = (0, 0)

    if not do_mask:
        return {
            "inputs": " ".join(prompt),
            "pre_range": pre_range,
            "hyp_range": hyp_range,
        }

    return {
        "inputs": " ".join(prompt),
        "target_token_idx": target_token_idx,
        "pre_range": pre_range,
        "hyp_range": hyp_range,
    }


def load_create_context_function(task: str):
    f_mapper = {
        "xnli": create_xnil_context,
        "marc2": create_marc2_context,
    }
    assert task in f_mapper, f"Sorry, {task} context function is not implemented..."
    return f_mapper[task]


def load_create_prompt_function_per_label(task: str):
    f_mapper = {
        "xnli": create_xnil_prompt_per_label,
        "marc2": create_marc2_prompt_per_label,
    }
    assert task in f_mapper, f"Sorry, {task} prompt function is not implemented..."
    return f_mapper[task]


def load_create_prompt_function(task: str):
    f_mapper = {"xnli": create_xnil_prompt, "marc2": create_marc2_prompt}
    assert task in f_mapper, f"Sorry, {task} prompt function is not implemented..."
    return f_mapper[task]


def label_tokenize_function(dataset: Dataset, tokenizer: AutoTokenizer, label):
    inputs = dataset[label]
    tokenized_ids = tokenize_function(inputs=inputs, tokenizer=tokenizer)
    input_ids, masks = tokenized_ids["input_ids"], tokenized_ids["attention_mask"]
    return input_ids, masks


def tokenize_function(example, tokenizer, max_length: int = 512, do_mask: bool = False):
    outputs = tokenizer(example["inputs"])
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]

    if not do_mask:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    target_token_idx = example["target_token_idx"]
    token_idxs = outputs.word_ids(0)
    masked_inputs = [
        input_id if token_idx != target_token_idx else tokenizer.mask_token_id
        for input_id, token_idx in zip(input_ids, token_idxs)
    ]

    assert len(input_ids) == len(attention_mask) == len(masked_inputs)

    return {
        "input_ids": masked_inputs,
        "attention_mask": attention_mask,
        "labels": input_ids,
    }


def batch_tokenize_function(
    example,
    tokenizer,
    max_length: int = 512,
    do_mask: bool = False,
    input_column: str = "inputs",
):
    outputs = tokenizer.batch_encode_plus(
        example[input_column],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]

    if not do_mask:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    masked_inputs = []
    for i in range(len(input_ids)):
        target_token_idx = example[f"{input_column}_target_token_idx"][i]
        token_idxs = outputs.word_ids(i)
        masked_input = [
            (
                input_id.item()
                if token_idx != target_token_idx
                else tokenizer.mask_token_id
            )
            for input_id, token_idx in zip(input_ids[i], token_idxs)
        ]
        masked_inputs.append(masked_input)

    return {
        "input_ids": torch.tensor(masked_inputs),
        "attention_mask": attention_mask,
        "labels": input_ids,
    }


def tokenize_function_ft(
    example,
    tokenizer,
    task: str,
    max_length: int = 512,
):
    if task == "xnli":
        return tokenizer(
            example["premise"],
            example["hypothesis"],
        )


def text_with_prompt(text: str, prompt: str, translate_args: Dict[str, Any]) -> str:
    """
    Concatenate the text with the prompt and the eos_token.

    Args:
    - text: A string representing the text to be concatenated.
    - prompt: A string representing the prompt to be concatenated.
    - translate_args: A dictionary containing the translation configurations.

    Returns:
    - A string representing the concatenated text with the prompt and the eos_token.
    """
    return f"{prompt} {text}{translate_args['eos_token']}English:"


lang_names = {
    "ru": "Russian",
    "zh": "Chinese",
    "es": "Spanish",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "te": "Telugu",
    "sw": "Swahili",
    "eu": "Basque",
    "my": "Burmese",
    "en": "English",
}


def create_mt_context(
    src_lang: str,
    tgt_lang: str,
    src_texts: List[str],
    tgt_texts: List[str],
    eos_token: str,
) -> Dict[str, str]:
    prompt = ""
    for i in range(len(src_texts) - 1):
        prompt += f"{lang_names[src_lang]}: {src_texts[i]}{eos_token}"
        prompt += f"{lang_names[tgt_lang]}: {tgt_texts[i]}{eos_token}"

    prompt += f"{lang_names[src_lang]}: {src_texts[i + 1]}{eos_token}"
    prompt += f"{lang_names[tgt_lang]}: "

    return prompt


def create_lm_context(
    texts: List[str],
    lang: str,
    eos_token: str,
) -> Dict[str, str]:
    prompt = ""
    for i in range(len(texts)):
        prompt += f"{lang_names[lang]}: {texts[i]}{eos_token}"
    prompt += f"{lang_names[lang]}: "
    return prompt
