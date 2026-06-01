# Topology-Preserving State-Space Diagnosis of Weather Forecast Trajectories

This repository provides source code and public reproduction materials for the lead-time and valid-time trajectory diagnosis experiments described in the accompanying manuscript.

The framework diagnoses weather forecast evolution in a learned topology-preserving state space. Instead of relying only on pointwise error metrics such as RMSE, the workflow compares forecast trajectories against reference trajectories in a low-dimensional state space constructed from multivariate atmospheric fields.

## Overview

The public reproduction workflow focuses on two main diagnostic tasks:

1. **Lead-time trajectory diagnosis**
   Evaluates forecast trajectories initialized from selected dates and evolved over multiple lead times.

2. **Valid-time trajectory diagnosis**
   Evaluates forecasts from different initialization dates that verify at the same target time.

The full source code for feature extraction, contrastive representation learning, and state-space construction is included for transparency. However, the public reproduction workflow does not require the original raw meteorological fields. Instead, it uses processed trajectory outputs and precomputed RMSE curves included in this repository.

## Archived code and reproduction materials

The source code, trained model files, processed trajectory outputs, precomputed RMSE curves, and configuration files required for the public reproduction workflow are included in this repository.

A frozen version of this repository is archived on Zenodo:

```text
Zenodo DOI: https://doi.org/10.5281/zenodo.20483795
```

The archived software package includes the reproduction materials contained in the GitHub release, including:

```text
out_test_features/
processed_outputs/
trained_models/
configs/
```

No separate raw-data archive is provided. The raw IFS analysis and forecast fields used in the full experiments were obtained through institutional access and cannot be publicly redistributed. The public reproduction workflow therefore uses processed trajectory outputs and precomputed RMSE curves rather than the restricted raw gridded fields.

## Preparing the repository

Clone or download this repository.

The public reproduction materials are already included in the repository and its archived release. The default workflow uses existing processed outputs, so raw input data are not required for reproducing the archived trajectory figures and RMSE curves.

The key directories are:

```text
out_test_features/
processed_outputs/
trained_models/
configs/
```

The YAML files under `configs/` define the default dates, output paths, and model selections. In most cases, users only need to change the model list if they want to display different model configurations.

## Reproducing lead-time trajectory diagnosis

Run:

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml
```

By default, this command uses existing processed outputs and precomputed RMSE curves. It does not require raw gridded input fields.

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

```yaml
models:
  model_list: [1, 4, 7]
```

To display all nine model configurations:

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --model-list 1 2 3 4 5 6 7 8 9
```

To display a different subset of models:

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --model-list 2 5 8
```

## Reproducing valid-time trajectory diagnosis

Run:

```bash
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml
```

By default, this command uses existing processed outputs and precomputed RMSE curves. It does not require raw gridded input fields.

The default sample dates are defined in `configs/paper_valid_time.yaml`:

```yaml
dates:
  include_dates:
    - "2025011500"
    - "2025041500"
    - "2025071500"
    - "2025101500"
```

The default representative models are:

```yaml
models:
  model_list: [1, 4, 7]
```

To display all nine model configurations:

```bash
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml --model-list 1 2 3 4 5 6 7 8 9
```

To display a different subset of models:

```bash
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml --model-list 3 6 9
```

## Model index

The model indices used by `--model-list` correspond to:

```text
1: fnet_ifs
2: fnet_kim
3: fnet_um
4: grph_ifs
5: grph_kim
6: grph_um
7: pang_ifs
8: pang_kim
9: pang_um
```

The same indices are used for both lead-time and valid-time trajectory diagnosis.

## Important note on figure navigation

The reproduction scripts generate figures sequentially for the configured dates.

After a figure is displayed, click inside the most recently generated figure window to continue to the next date. The script waits for this interaction before moving on to the next trajectory example.

If the script appears to pause after producing a figure, activate the latest figure window and click inside it.

## Configuration files

The main public configuration files are:

```text
configs/paper_lead_time.yaml
configs/paper_valid_time.yaml
```

The most important setting for public reproduction is the model list:

```yaml
models:
  model_list: [1, 4, 7]
```

This can also be overridden from the command line:

```bash
python -m feat_space_analysis.cli lead --config configs/paper_lead_time.yaml --model-list 1 4 7
python -m feat_space_analysis.cli valid --config configs/paper_valid_time.yaml --model-list 1 4 7
```

The default execution mode is designed to show existing processed results. Since the public repository does not include raw gridded input fields, users normally do not need to change the `show_only` setting.

## Notes on raw-data mode

The public repository does not include the restricted raw IFS analysis and forecast fields. Therefore, users should normally run the scripts in the default show-only workflow.

Options that rerun inference or recompute diagnostics from raw gridded fields are intended only for internal use, or for users who have separately prepared the required raw input fields in the expected directory structure. These options are not required for the public reproduction workflow.

## Notes on reproducibility

The default public workflow is designed to reproduce the diagnostic trajectory figures, RMSE curves, and related outputs using archived processed results, precomputed RMSE curves, and trained models.

The repository also contains source code for the broader framework, including feature extraction, representation learning, and state-space construction. However, full reprocessing from the original meteorological fields is not part of the default public workflow, because the raw IFS analysis and forecast data used in the full experiments were obtained through institutional access and cannot be publicly redistributed.

To support reproducibility within these restrictions, the repository includes processed trajectory outputs and precomputed RMSE curves rather than the restricted raw gridded fields.

## Citation

If you use this code or reproduction package, please cite the accompanying paper and the archived Zenodo record.

```text
Paper:
Kim, H. and Cho, J. H.: Topology-Preserving State Space Representation for Diagnosing Weather Forecast Models, submitted to Geoscientific Model Development, 2026.

Code and reproduction package:
https://doi.org/10.5281/zenodo.20483795
```

## License

This repository is distributed under the terms of the license specified in `LICENSE`.

## Contact

For questions about the code or reproduction workflow, please contact:

```text
Hyoungnyoun Kim
nyoun [at] koera [dot] kr
```

