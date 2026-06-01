# Topology-Preserving State-Space Diagnosis of Weather Forecast Trajectories

This repository provides source code for reproducing the lead-time and valid-time trajectory diagnosis experiments described in the accompanying manuscript.

The framework diagnoses weather forecast evolution in a learned topology-preserving state space. Instead of relying only on pointwise error metrics such as RMSE, the workflow compares forecast trajectories against reference trajectories in a low-dimensional state space constructed from multivariate atmospheric fields.

## Overview

The public reproduction workflow focuses on two main diagnostic tasks:

1. **Lead-time trajectory diagnosis**
   Evaluates forecast trajectories initialized from selected dates and evolved over multiple lead times.

2. **Valid-time trajectory diagnosis**
   Evaluates forecasts from different initialization dates that verify at the same target time.

The full source code for feature extraction, contrastive representation learning, and state-space construction is included for transparency. However, the default reproduction workflow uses archived inference-ready input data and trained models, because rebuilding the full state space from the original meteorological datasets requires large external datasets and substantial computation.


## Archived data and trained models

The input data and trained model files required for reproduction are archived separately on Zenodo:
Zenodo DOI:  https://doi.org/10.5281/zenodo.20441860

The Zenodo archive is expected to include:

data/input/<tag_name>/Test_4var1lev/
2025011500_lead/, 2025041500_lead/, 2025071500_lead/, 2025101500_lead/
2025011500_valid/, 2025041500_valid/, 2025071500_valid/, 2025101500_valid/

The exact file organization may depend on the final Zenodo package. If the archive is extracted into the repository root, the default `paper_*.yaml` configuration files should work without major modification.


## Preparing the data

Download the Zenodo archive and extract it into the repository root.

After extraction, the following paths should exist:
data/input/<tag_name>/Test_4var1lev/

For local testing with a different directory structure, edit the corresponding YAML files under `configs/`.

The most important path settings are:

paths:
  input_root_dir: data/input
  input_subdir: Test_4var1lev
  processed_root_dir: out_test_features

These paths are interpreted as follows:


Inference input:
{input_root_dir}/{tag_name}/{input_subdir}/{target_date}{suffix}

Processed output:
{processed_root_dir}/{tag_name}/{target_date}{suffix}


For lead-time trajectory diagnosis, the suffix is empty:

```yaml
folders:
  target_folder_suffix: "_lead"
```

For valid-time trajectory diagnosis, the current suffix is:

```yaml
folders:
  target_folder_suffix: "_valid"
```

If the valid-time folders are renamed later, for example to `_valid`, only the YAML file needs to be updated.

## Reproducing lead-time trajectory diagnosis

Run:

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml
```

By default, this command uses existing processed outputs in:

```text
out_test_features/<tag_name>/<target_date>_lead
```

The default sample dates are defined in `configs/paper_lead_time.yaml`:

```yaml
dates:
  include_dates:
    - "2025011500"
    - "2025041500"
    - "2025071500"
    - "2025101500"
```

The default representative models are:

models:
  model_list: [1, 4, 7]

To run the same lead-time workflow with all nine model configurations:

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --model-list 1 2 3 4 5 6 7 8 9
```

To rerun inference from the prepared input folders instead of only showing existing processed results:

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --no-show-only
```

In this case, the command uses input data from:

```text
data/input/<tag_name>/Test_4var1lev/<target_date>_lead
```

and writes processed outputs to:

```text
out_test_features/<tag_name>/<target_date>_lead
```

## Reproducing valid-time trajectory diagnosis

Run:

```bash
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml
```

By default, this command uses existing processed outputs in:

```text
out_test_features/<tag_name>/<target_date>_valid
```

The default sample dates are defined in `configs/paper_valid_time.yaml`:

```yaml
dates:
  include_dates:
    - "2025011500"
    - "2025041500"
    - "2025071500"
    - "2025101500"
```

To run the same valid-time workflow with all nine model configurations:

```bash
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml --model-list 1 2 3 4 5 6 7 8 9
```

To rerun inference from the prepared input folders:

```bash
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml --no-show-only
```

In this case, the command uses input data from:

```text
data/input/<tag_name>/Test_4var1lev/<target_date>_valid
```

and writes processed outputs to:

```text
out_test_features/<tag_name>/<target_date>_valid
```

## Configuration files

The main public configuration files are:

```text
configs/paper_lead_time.yaml
configs/paper_valid_time.yaml
```


## Command-line options

The command-line arguments override the YAML settings.

Example: use a different model list.

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --model-list 1 2 3
```

Example: rerun inference instead of only showing existing results.

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --no-show-only
```

Example: explicitly use existing processed results.

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --show-only
```

Example: change the valid-time suffix from `_target` to `_valid`.

```bash
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml --target-folder-suffix _valid
```

## Notes on reproducibility

The default public workflow is designed to reproduce the diagnostic trajectory figures and related outputs using archived input data, trained models, and processed feature-space results.

The repository also contains source code for the broader framework, including feature extraction, representation learning, and state-space construction. However, full re-training from the original meteorological data is not part of the default public workflow, because it requires large external datasets and substantial computation.

## Citation

If you use this code or data, please cite the accompanying paper and the archived Zenodo record.

```text
Paper:
<TO_BE_ADDED>

Code and data:
<TO_BE_ADDED>
```

## License

This repository is distributed under the terms of the license specified in `LICENSE`.

## Contact

For questions about the code or reproduction workflow, please contact:

```text
<TO_BE_ADDED>
```
