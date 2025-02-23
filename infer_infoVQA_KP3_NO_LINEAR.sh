cd VLMEvalKit

CUDA_VISIBLE_DEVICES=2,3 \
HF_HUB_OFFLINE=1 \
AUTO_SPLIT=1 \
LS=0.6 \
KP=0.3 \
IL=False \
python run.py \
--data InfoVQA_VAL \
--model Qwen2-VL-7B-Instruct \
--verbose \
--work-dir results_InfoVQA_VAL_LS6_KP3 \