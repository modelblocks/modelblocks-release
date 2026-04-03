# Surprisal from Larger Transformer-based Language Models Predicts fMRI Data More Poorly

Code and instructions for replicating experiments from [Surprisal from Larger Transformer-based Language Models Predicts fMRI Data More Poorly](https://aclanthology.org/2026.eacl-short.11/) (Lin and Schuler, 2026).

Please contact Yi-Chien Lin ([lin.4434@osu.edu](mailto:lin.4434@osu.edu)) if you have any questions.

## Setup
### Conda Environment Dependencies
Following are the versions of the major dependencies used in this work:
* [PyTorch](https://pytorch.org/): v2.5.1
* [HuggingFace Transformers](https://huggingface.co/docs/transformers/installation): v4.48.2

### Dataset Processing and Regession
The scripts used for running experiments in this study depend on other scripts from the [modelblocks-release](https://github.com/modelblocks/modelblocks-release) repository. Please clone the repository to ensure you have all necessary scripts for processing the datasets, colleting LM surprisal estimates, and running the linear mixed-effects (LME) regression experiments.

## Scripts
The end-to-end scripts for processing the datasets, collecting LM surprisal estimates, and running LME regression are located under the `scripts/` subdirectory. To run an experiment using surprisal estimtes from a specific LM, simply pass the model variant as an argument to the bash script. Following are the model variants evaluated in this study:
```
gpt2 family:
gpt2, gpt2-medium, gpt2-large, gpt2-xl

gpt-neo family:
gpt-neo-125m, gpt-neo-1300m, gpt-neo-2700m, gpt-j-6000m, gpt-neox-20000m

opt family:
opt-125m, opt-350m, opt-1300m, opt-2700m, opt-6700m, opt-13000m, opt-30000m, opt-66000m
```

Following are the examples for getting the surprisal estimates from gpt2-small and running the regression experiments for both corpora:

* Natural Stories fMRI:
    ```
    naturalstoriesfmri_get_loglik.sh gpt2
    ```
* Pereira fMRI:
    ```
    pereirafmri_get_loglik.sh gpt2
    ```

The bash scripts include a series of `make` commands. Any missing intermediate files will be identified and generated during the execution of the scripts. Detailed breakdown of the steps can be found in the bash scripts. The formulas used in this study are specified in `fmri.lmerform` (for Natural Stories fMRI) and `fmri-sent.lmerform` (for Pereira fMRI). If you wish to use your own formulas, you can do this by modifying the `.lmerform` files.