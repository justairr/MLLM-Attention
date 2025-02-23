cd VLMEvalKit

CUDA_VISIBLE_DEVICES=4,5 \
HF_HUB_OFFLINE=1 \
AUTO_SPLIT=1 \
LS=0.6 \
KP=1.0 \
python run.py \
--data DocVQA_VAL \
--model Qwen2-VL-7B-Instruct \
--verbose \
--work-dir results_DocVQA_VAL_KP10 \