## Code Base and Acknowledgement

This repository is based on the official implementation of **SAIL: Self-Improving Efficient Online Alignment of Large Language Models**.
We build upon the original SAIL codebase and introduce additional engineering refinements and extensions, 
including a lightweight integration with HuggingFace's `DPOTrainer`, improved experiment configuration, 
and extended evaluation pipelines.

The original SAIL repository can be found at:
[https://github.com/johnding1996/SAIL]

### Step 1: Filling YAML Configs
1. We need HuggingFace Hub and WanDB to manage experiments. Please fill in `./configs/services/hugggingface.yaml` and `./configs/services/wandb.yaml` with your acconut info.
2. We need OpenAI api to evaluate models. Please fill in `./configs/services/openai.yaml` with your account info.
3. We use a HuggingFace Space App to retrieve and review results. Please fill in `./viewer/.env` with your account info.

### Step 2: Installation
1. With `python 3.10.*` and `CUDA 12.*` installed. You can run `python install -e .` to install this package called `cdpo`.

### Step 3: Config the Experiments to Run
1. Fill in or modify `./configs/tasks.yaml` for the set of experiments to run.

### Step 4: Run Experiments
1. Execute the corresponding scripts to launch experiments on different model backbones:
   - `./run_L8B.sh` for LLaMA-3 (8B)
   - `./runQwen.sh` for Qwen 0.5B models
   - `./run_Phi.sh` for Phi 3B models
