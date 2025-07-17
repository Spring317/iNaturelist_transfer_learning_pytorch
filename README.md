# Edge-Aware Fine-Grained Classification using Transfer Learning implemented in Pytorch

## Table of contents

1. [Overview](#overview)
2. [Credits](#credits)
3. [Dependencies and Installation](#dependencies-and-installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Project Structure](#project-structure)
6. [Key Design Choices](#key-design-choices)

## Overview
This project is the continuation of this [repo](https://github.com/HoangPham6337/iNaturelist_transfer_learning_pytorch), with much focused on training on a specific subset (Insecta dataset in this case). We explores the effectiveness of caching mechanism on machine learning. We aim to optimize fine-grained species classification for edge devices.

### Objectives
- [x] Fine-tuned the model on a regional species subset.
<!-- - [x] Implement an 'Other' classification for non-dominant species. -->
<!-- - [x] Create a custom `logits` layer (final characterization) using `logsumexp` that can vary the output based on dominant threshold.  -->
<!-- - [x] Prune layer by layer -->
<!-- - [x] Prune globally -->
<!-- - [x] Prune based on the activated feature maps. The experiment result is available at [activation_based_pruning_exp branch](https://github.com/HoangPham6337/iNaturelist_transfer_learning_pytorch/tree/activation_based_pruning_exp) -->
- [ ] Analyze the feature maps of the model if fine-tuning proves insufficient
- [ ] Improve real-time performance & efficiency of classification models

### Why Edge computing?
In real-world scenario, deploying large-scale models on edge devices (RaspberryPi, IoT devices, ...) is challenging due to:
- Limited computational power
- Lower memory availability
- Network independence

By fine-tuning the model with a focused dataset and optimizing its feature representations, we aim to enhance model performance while minimizing resource consumption.

## Credits 
Authors of the original works this project based on:

- [Yin Cui](http://www.cs.cornell.edu/~ycui/)
- [Yang Song](https://ai.google/research/people/author38269)
- [Chen Sun](http://chensun.me/)
- Andrew Howard
- [Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)
-  [Pham Xuan Hoang](https://hoangpham6337.github.io/portfolio/)

**This project is based on the original work from:**
- [Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning (CVPR 2018)](https://arxiv.org/abs/1806.06193)
- [Original GitHub Repository](https://github.com/richardaecn/cvpr18-inaturalist-transfer)

## Dependencies and Installation

This codebase uses `Pytorch 2.6.0` as its backbone.
Setting up is fairly straight forward, you can install all the dependencies through:
```python
   conda env create --name training --file=environment.yml
   conda activate training
```

The `dataset_builder` install instructions can be found by going to this [repo](https://github.com/HoangPham6337/iNaturelist_dataset_builder).

## Dataset Preparation
**This project uses a subset of iNaturelist 2017 combined with species from Haute-Garonne. The dataset must be manually downloaded and processed before training.**

### Dataset Preparation
We provide a modular and automated pipeline using the `dataset_builder` package. Configuration is handled via [config.yaml](config.yaml).

### Steps (automated by [dataset_orchestor.py](dataset_orchestrator.py))

1. Crawl species from iNaturalist (Haute-Garonne region)
2. Analyze dataset structure (class/species breakdown, image counts)
3. Cross-reference species between source and regional datasets
4. Copy matched species into a new dataset to create regional datasets
5. Label species based on dominant threshold
5. Generate train/validation manifests 
10. Produce visualizations (bar charts, CDF, PPF, Venn diagrams)

If any operations fails, a `FailedOperation` is rased and the script will:
- Print a traceback
- Exit gracefully

Fist of all please run [dataset_orchestor.py](dataset_orchestrator.py)
```bash
python3 dataset_orchestrator.py
```
### Configuration file: [config.yaml](config.yaml)
This file can be created automatically through the `create_interactive_config.py`.

Normally, you don't want to touch anything in here if you want to recreate what I did. But in case you do want to conduct your test on your own, these are some modifications you should     make:
```yaml
global:
  included_classes: ["Aves", "Insecta"]  # Species class to analyze
  verbose: false  # Print extra debugging info
  overwrite: false  # Overwrite existing file

paths:
  src_dataset: "./data/inat2017"  # Source dataset
  dst_dataset: "./data/haute_garonne"  # Target dataset
  web_crawl_output_json: "./output/haute_garonne.json"  # Path to save crawl result 
  output_dir: "./output"  # Path to save all JSON files

web_crawl:
  total_pages: 104
  base_url: "https://www.inaturalist.org/check_lists/32961-Haute-Garonne-Check-List?page="
  delay_between_requests: 1

train_val_split:
  train_size: 0.8
  random_state: 42
  dominant_threshold: 0.5
```

Output:
- `*_species.json`: class → list of species
- `*_composition.json`: class/species → {species: count}
- `matched_species.json`: Cross-reference results
- `train.parquet`, `val.parquet`, `dataset_manifest.parquet`: Data splits
- `plots/`: CDF, PPF, Venn diagrams, class bar charts

## Piping models inference:
1. As an effort of optimizing the model in both accuracy and efficiency, we introduce the model piping strategy:

* **Piping**: Combining several small models in front of the main model to filter out non-dominant species, reducing the computational load on the main model:
  - **Cache model**: A lightweight model (**mcunet**) that classifies dominant species. These models (1 for now), filter out the species with higher representations in the dataset else returning "Others".
  - The "Others" class is then mapped to teh full dataset and being processed by the main model to return the final classification.
  - **Main Model**: A more complex model that trained with the whole dataset, taking in charge of predicting every classes.
  - Why we are not excluding the classes that are predicted by the cache model? Because everyone deserve a second chance :D. If the cache model predict one dominant class as others, the main model will correct it, increase the accuracy.


2. Inferencing steps:
 - Run this command the run the inference: 

```bash
python FullPipelineMonteCarloSimulation.py --model both --input_size 160 --runs 10 --samples 500 --out_prefix myexp
```

Feel free to change options with --help flags.
## Project Structure

All pipeline core functions has been implemented in `pipeline/`

### Training Pipeline
- `train_single.py`: Train a baseline MobileNetV3 model (with optional experimental hyperparameter tuning).
- `train_multiple.py`: Retrains pruned models (from `models/`) with different dominant thresholds

### Pruning Pipeline
- `prune.py`: Applies structure **global** (with isomorphic) and **layer by layer** (with isomorphic) pruning using `torch-pruning` ([link](https://github.com/VainF/Torch-Pruning))
- `experiment_prune.py`: Demonstrates layer-specific pruning with `pruning_ratio_dict`.
- We uses Magnitude Pruning (L1 norm).

### Feature Map Analysis
- `feature_maps_extractor.py`: Extracts shape + memory shape of feature maps.
- `feature_maps_analysis.py`: Heatmap visualization across pruning levels and dominance thresholds.

### MACs + Parameter Profiling
- `mac_cal.py`: Uses `thop.profile()` to calculate MACs and parameters of pruned models.

### Monte Carlo Simulation
- `MonteCarloBenchmark.py`: Baseline test that groups non-dominant species into an "Other" class for practical deployment settings.
  - Repeatedly samples species and evaluates ONNX models.
  - Evaluates communication rate, false positive rate (FPR), confusion matrix.
  - Evaluates robustness across random compositions and species coverage.

### Post-hoc Evaluation
- `post_hoc_approach.py`: Proof-of-concept that groups non-dominant species into an "Other" class for practical deployment settings.

## Key Design Choices
- Dominant Species Filtering: Allows dynamic focus on high-frequency species.
- Model Compression: Achieved via structured pruning with retraining for accuracy recovery.
- Robust Evaluation: Monte Carlo simulations simulate field deployment by testing varying species distributions.
- Interpretability: Feature map memory footpring and sparsity metrics are visualized layer-by-layer.
