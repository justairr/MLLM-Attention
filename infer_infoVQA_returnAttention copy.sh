cd VLMEvalKit

CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
AUTO_SPLIT=1 \
python run.py \
--data InfoVQA_VAL \
--model InternVL2-8B \
--verbose \
--work-dir results_InfoVQA_VAL_internvl_selfmade \