# Code for our paper "Pruning Multilingual Large Language Models for Multilingual Inferece"

## Environment
- OS: Ubuntu 20.04.6 LTS
- Python: 3.8.5
- CUDA: 12.1

## Install libraries

`pip install -r requirements.txt`

## Usage

### Plot magnitudes of features like Figure 2

- Plot magnitudes of features when inputting few-shot monolingual demonstrations
```
cd experiments/analysis
bash lm_analysis.sh [GPU_NUMBER] [MODEL_NAME] [LANG_NAME]
```

`[GPU_NUMBER]` refers to an identifier of the GPU you wish to use (e.g., `0`). `[MODEL_NAME]` refers to a model name you intend to use (e.g., `facebook/xglm-2.9B`). `[LANG_NAME]` refers to language whose features you would like to examine (e.g., `en`, `fr`, etc.).

- Plot magnitudes of features when inputting few-shot translation demonstrations
```
cd experiments/analysis
bash mt_analysis.sh [GPU_NUMBER] [MODEL_NAME] [SRC_LANG_NAME] [TGT_LANG_NAME]
```

`[SRC_LANG_NAME]` and `[TGT_LANG_NAME]` refer the source and target languagers, respectively. 

The bar graph (and the csv file that are stored the hidden states) will be saved in `outputs/[MODEL_NAME]/analysis/[LANG_NAME]` or `outputs/[MODEL_NAME]/analysis/[SRC_LANG_NAME]-[TGT_LANG_NAME]`.


### Plot overlap heatmaps like Figures 1 and 3

```
cd src
python overlap_analysis.py \
  --root_dir ../outputs/[MODEL_NAME]/analysis \
  --langs [LANG_NAMES] \
  --save_dir [PATH_TO_SAVE_FIG (e.g. outputs/[MODEL NAME]/analysis/overlap)] \
  --ratio [RATIO (e.g. 0.3)]
```

Set language names for overlap measurement (e.g., `en zh`) in `[LANG_NAMES]`. `[PATH_TO_SAVE_FIG]` specifies a directory where the heatmaps will be saved (e.g., `outputs/[MODEL_NAME]/analysis/overlap`). `[Ratio]` determines a ratio (e.g., `0.3`) of top and bottom features used for the overlap measurement.

To generate heatmaps similar to Figures 1 and 3, run the experiments with the languages en, fr, es, zh, hi, and sw to obtain their hidden state features, and specify `en fr es zh hi sw` as the argument for `--langs`. 

### Pruning Model

```
cd experiments/prune
bash prune.sh [GPU_NUMBER] [MODEL_NAME] [PRUNING_RATIO (e.g. 0.3)] [DATA_TYPE (e.g. mono, mt, mt_code)] [SRC_LANG_NAME] [TGT_LANG_NAME]
```

`[PRUNING_RATIO]` specifies a pruning ratio (e.g., `0.3`). If you wish to prune the model using monolingual demonstrations, translation demonstrations, or a combination of translation and programming language demonstrations, set `[Data_Type]` to `mono`, `mt`, or `mt_code`, respectively. `[SRC_LANG_NAME]` refers to a source language of demonstrations used for pruning (e.g., `en`, `fr`, etc.). Additionally, if you are pruning the model using translation demonstrations, please specify a target language by setting `[TGT_LANG_NAME]`. The pruned model will be saved in the directory `outputs/[MODEL_NAME]/[SRC_LANG_NAME] (or [SRC_LANG_NAME]-[TGT_LANG_NAME])/sparsity_type=unstructured_ratio=[PRUNING_RATIO]_shots=4_samples=100`.

### Perform XX-English Translation like Table 1

- Translation
```
cd experiments/translate
bash translate.sh [GPU_NUMBER] [MODEL_NAME] [PRUNED_MODEL_PATH (if you experiment on original model, you set [MODEL_NAME])]
```

`[PRUNED_MODEL_PATH]` refers to the path of the pruned model (e.g., `outputs/[MODEL_NAME]/[SRC_LANG_NAME] (or [SRC_LANG_NAME]-[TGT_LANG_NAME])/sparsity_type=unstructured_ratio=[PRUNING_RATIO]_shots=4_samples=100`). If you are conducting experiments on the original model, set `[PRUNED_MODEL_PATH]` to the name of the original model (e.g., `facebook/xglm-2.9B`).

The results will be saved in `outputs/[MODEL_NAME]/translation/xnli/xnli.[SRC_LANG_NAME].test.tsv`, where `[SRC_LANG_NAME]` denotes the source language.

- Significance test
```
cd experiments/translate
bash siginificant_test.sh [HYP_PATH1] [HYP_PATH2]
```

Set the paths to the hypothesis files (i.e., the aforementioned TSV files) from each system to `[HYP_PATH1]` and `[HYP_PATH2]`.

### Perform Zero-shot Learning like Tables 2 and 3

- XNLI task
```
cd experiments/evaluate
bash xnli.sh [GPU_NUMBER] [MODEL_NAME_OR_PATH] [LANG_NAME]
```

`[MODEL_NAME_OR_PATH]` refers to either the name of the original model or the path to the pruned model. `[LANG_NAME]` refers to the language of the evaluation dataset (e.g., `en`, `fr`, etc.).

- MARC task

```
python data/download.py --save_dir data
cd experiments/evaluate
bash marc2.sh [GPU_NUMBER] [MODEL_NAME_OR_PATH] [LANG_NAME]
```

The probabilities for each candidate label will be saved in `[MODEL_NAME_OR_PATH]/[xnli or marc2]/[LANG_NAME].test.npy`

- Significance test
```
cd experiments/evaluate
bash significance_test.sh [PROB_PATH1] [PROB_PATH2] [TASK (i.e. xnli or marc2)] [LANG_NAME]
``` 
Where `[PROB_PATH1]` and `[PROB_PATH2]` denote the npy files mentioned above for each system.

### Measure RankC like Table 4

```
python src/evaluate_consistency.py \
    --root_dir [PATH_TO_PROBS (e.g. [MODEL_NAME_OR_PATH]/xnli)] \
    --langs [LANG_NAMES (e.g. fr es zh)]
```

`[PATH_TO_PROBS]` refers to the directory where the probability data is stored (e.g., `[MODEL_NAME_OR_PATH]/xnli`). Specify the languages for which you would like to measure consistencies with English in `[LANG_NAMES]` (e.g., `fr es zh`).
