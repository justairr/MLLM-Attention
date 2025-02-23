cd VLMEvalKit

CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
AUTO_SPLIT=1 \
LS=0.6 \
KP=1.0 \
python run.py \
--data InfoVQA_VAL \
--model Qwen2-VL-7B-Instruct \
--verbose \
--work-dir results_InfoVQA_VAL_KP10 \