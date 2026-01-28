ROOT="$(cd "$(dirname "$0")" && pwd)"
cd ~/work/SAIL
conda activate SAIL_LLM


# 1e-3KL Pm3B U10

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes 4\
  --config_file $ROOT/configs/accelerate/sft/zero2.yaml \
  $ROOT/pipelines/sft.py \
  --model Pm3B \
  --dataset U10 \
  --tag Pm3B1e-3KL \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1


CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --config_file $ROOT/configs/accelerate/dpo/zero2.yaml \
  $ROOT/pipelines/dpo.py \
  --pipeline DPR \
  --model Pm3B \
  --dataset U10 \
  --tag Pm3B1e-3KL \
  --beta 0.10 \
  --g 0.30 \
  --gamma 0.30 \
  --revkl \
  --revkl_coef 1e-3 \
  --revkl_on both \
  --model_max_length 512 \
  --max_steps 100 \
  --warmup_steps 10

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes 4\
  --config_file $ROOT/configs/accelerate/generate/ddp.yaml \
  $ROOT/pipelines/generate.py \
  --run DPR_Pm3B_U10_beta0.10g0.30gamma0.30 \
  --tag Pm3B1e-3KL \
  --use_flash_attn false \
  --eval_limit 100 


CUDA_VISIBLE_DEVICES=0,1,2,3 \
TOKENIZERS_PARALLELISM=false \
accelerate launch --num_processes 4\
  --config_file $ROOT/configs/accelerate/evalreward/ddp.yaml \
  $ROOT/pipelines/evalreward.py \
  --run DPR_Pm3B_U10_beta0.10g0.30gamma0.30 \
  --tag Pm3B1e-3KL \
  --every_k 2 \
  --per_device_evalreward_batch_size 2 \
  --model_max_length 512 \
  --padding true \
  --truncation true

  python $ROOT/pipelines/evalgpt.py \
  --run DPR_Pm3B_U10_beta0.10g0.30gamma0.30 \
  --tag Pm3B1e-3KL


## DPR Pm3B U10
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --config_file $ROOT/configs/accelerate/sft/zero2.yaml \
  $ROOT/pipelines/sft.py \
  --model Pm3B \
  --dataset U10 \
  --tag Pm3B \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --config_file $ROOT/configs/accelerate/dpo/zero2.yaml \
  $ROOT/pipelines/dpo.py \
  --pipeline DPR \
  --model Pm3B \
  --dataset U10 \
  --tag Pm3B \
  --beta 0.10 \
  --g 0.30 \
  --gamma 0.30 \
  --model_max_length 512 \
  --max_steps 100 \
  --warmup_steps 10

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes 4\
  --config_file $ROOT/configs/accelerate/generate/ddp.yaml \
  $ROOT/pipelines/generate.py \
  --run DPR_Pm3B_U10_beta0.10g0.30gamma0.30 \
  --tag Pm3B\
  --use_flash_attn false \
  --eval_limit 100 


CUDA_VISIBLE_DEVICES=0,1,2,3 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file $ROOT/configs/accelerate/evalreward/ddp.yaml \
  $ROOT/pipelines/evalreward.py \
  --run DPR_Pm3B_U10_beta0.10g0.30gamma0.30 \
  --tag Pm3B \
  --every_k 2 \
  --per_device_evalreward_batch_size 2 \
  --model_max_length 512 \
  --padding true \
  --truncation true

  python $ROOT/pipelines/evalgpt.py \
  --run DPR_Pm3B_U10_beta0.10g0.30gamma0.30 \
  --tag Pm3B

