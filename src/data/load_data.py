import random
from typing import List

from datasets import DatasetDict, load_dataset

lang_codes = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "ru": "rus_Cyrl",
    "sw": "swh_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    "en": "eng_Latn",
}

lang_names = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "en": "English",
}


def get_dataset(task_name: str, lang1: str, lang2: str = None):
    if task_name == "mt":
        return get_mt_dataset(lang1=lang1, lang2=lang2)
    elif task_name == "lm":
        return get_lm_dataset(lang=lang1)


def get_mt_dataset(lang1: str, lang2: str) -> DatasetDict:
    dataset = DatasetDict()
    dataset[lang1] = load_dataset("facebook/flores", lang_codes[lang1], split="dev")
    dataset[lang2] = load_dataset("facebook/flores", lang_codes[lang2], split="dev")
    return dataset


def get_lm_dataset(lang: str) -> DatasetDict:
    dataset = DatasetDict()
    dataset[lang] = load_dataset("facebook/flores", lang_codes[lang], split="dev")
    return dataset


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    shots=4,
    max_length=512,
    src_lang=None,
    tgt_lang=None,
    tokenizer=None,
):
    trainloader = []
    data_types = []
    if "mt" in name:
        trainloader += get_mt(
            nsamples,
            seed,
            shots=shots,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        data_types += ["mt" for _ in range(nsamples)]
    if "mono" in name:
        trainloader += get_mono(
            nsamples,
            seed,
            shots=shots,
            lang=src_lang,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        data_types += ["mono" for _ in range(nsamples)]
    if "code" in name:
        trainloader += get_code(
            nsamples,
            seed,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        data_types += ["code" for _ in range(nsamples)]

    return trainloader, data_types


def get_code(
    nsamples: int,
    seed: int,
    tokenizer,
    max_length: int,
):
    # Load train and validation datasets

    dataset = load_dataset("thomwolf/github-python")
    dataset = dataset.shuffle(seed=seed)
    # Generate samples from training set
    trainloader = []
    for n in range(nsamples):
        prompt_text = dataset["train"][n]["content"]
        if n == 0:
            print("Example of input texts for activation")
            print(prompt_text)

        trainenc = tokenizer(
            [prompt_text],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inp = trainenc.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar, None, trainenc.attention_mask, None))

    return trainloader


def get_mono(
    nsamples,
    seed,
    shots,
    tokenizer,
    lang,
    max_length,
):
    # Load train and validation datasets

    dataset = DatasetDict()
    dataset[lang] = load_dataset(
        "facebook/flores",
        lang_codes[lang],
        split="dev",
    ).shuffle(seed=seed)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []

    eos_token = tokenizer.eos_token
    i = 0
    for n in range(nsamples):
        shot = 0
        prompt_text = ""
        while shot < shots:
            # if len(dataset[lang][i]["sentence"]) < 100:
            prompt_text += (
                f"{lang_names[lang]}: {dataset[lang][i]['sentence']}{eos_token}"
            )

            shot += 1
            i += 1

        # if version == "v1":
        #     prompt_text += f"{lang_names[lang]}:"

        if n == 0:
            print("Example of input texts for activation")
            print(prompt_text)

        trainenc = tokenizer(
            [prompt_text],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # tokenized_tokens = [tokenizer.decode([x]) for x in trainenc.input_ids[0]]
        # tokenized_tokens_wo_mask = trainenc.input_ids[trainenc.attention_mask.bool()]
        # target_token_index = detect_target_index(
        #     tokenized_tokens, f"{lang_names[lang]}:"
        # )
        # print(target_token_index)
        # print(f"{lang_names[lang]}:")
        # target_tokens = tokenized_tokens_wo_mask[target_token_index + 1 :]
        # token_num = len(target_tokens)
        token_num = 0
        target_token_index = 0
        inp = trainenc.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append(
            (inp, tar, token_num, trainenc.attention_mask, target_token_index)
        )

    return trainloader


def get_mt(
    nsamples,
    seed,
    shots,
    tokenizer,
    src_lang,
    tgt_lang,
    max_length,
):
    # Load train and validation datasets

    dataset = DatasetDict()
    dataset[src_lang] = load_dataset(
        "facebook/flores", lang_codes[src_lang], split="dev"
    )
    dataset[tgt_lang] = load_dataset(
        "facebook/flores", lang_codes[tgt_lang], split="dev"
    )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    eos_token = tokenizer.eos_token
    i = 0
    for n in range(nsamples):
        shot = 0
        prompt_text = ""
        while shot < shots:
            # if len(dataset[lang][i]["sentence"]) < 100:
            prompt_text += (
                f'{lang_names[src_lang]}: {dataset[src_lang][i]["sentence"]}{eos_token}'
            )
            prompt_text += (
                f'{lang_names[tgt_lang]}: {dataset[tgt_lang][i]["sentence"]}{eos_token}'
            )
            shot += 1
            i += 1

        # if version == "v1":
        #     prompt_text += (
        #         f'{lang_names[src_lang]}: {dataset[src_lang][i]["sentence"]}{eos_token}'
        #     )
        #     prompt_text += f"{lang_names[tgt_lang]}:"

        i += 1
        if n == 0:
            print("Example of input texts for activation")
            print(prompt_text)

        trainenc = tokenizer(
            [prompt_text],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # tokenized_tokens = [tokenizer.decode([x]) for x in trainenc.input_ids[0]]
        # tokenized_tokens_wo_mask = trainenc.input_ids[trainenc.attention_mask.bool()]
        # target_token_index = detect_target_index(
        #     tokenized_tokens, f"{lang_names[tgt_lang]}:"
        # )
        # target_tokens = tokenized_tokens_wo_mask[target_token_index + 1 :]
        # token_num = len(target_tokens)
        token_num = 0
        target_token_index = 0
        inp = trainenc.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append(
            (
                inp,
                tar,
                token_num,
                trainenc.attention_mask,
                target_token_index,
            )
        )
    return trainloader


def detect_target_index(tokenized_tokens: List[str], target_token: str):
    for i in range(len(tokenized_tokens) - 1, 0, -1):
        token1 = tokenized_tokens[i - 1]
        token2 = tokenized_tokens[i]

        if target_token in token1:
            return i - 1
        elif target_token in token1 + token2:
            return i
