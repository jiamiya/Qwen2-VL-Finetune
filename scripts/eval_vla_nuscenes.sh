
export PYTHONPATH=src:$PYTHONPATH
export TRITON_CACHE_DIR=/scratch/gilbreth/jmingyan/.triton/autotune
mkdir -p $TRITON_CACHE_DIR

python -u src/evaluation/eval_nuscenes_vla.py \
  --base_model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --ckpt_dir /scratch/gilbreth/jmingyan/project/Qwen2-VL-Finetune/output/testing_lora_qwen25vla_wrnc/checkpoint-1150 \
  --eval_json data/nuscenes_waypoint_short_prompt_val.json \
  --image_folder /scratch/gilbreth/cancui/data/nuscenes/full/samples/CAM_FRONT \
  --output_dir results/testing_lora_qwen25vla_wrnc/checkpoint-1150 \
  --batch_size 1 \
  --dtype bf16 \
  --device cuda \
  --horizons 1,2,3 \
  --dt 0.5