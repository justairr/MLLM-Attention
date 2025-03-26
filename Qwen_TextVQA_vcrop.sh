cd VLMEvalKit

DATASET=TextVQA_VAL #DocVQA_VAL A-OKVQA GQA_TestDev_Balanced
MODEL=Qwen2-VL-7B-Instruct
KP=1.0

CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 \
KP=${KP} \
python run.py \
--data ${DATASET} \
--model ${MODEL} \
--verbose \
--work-dir "../results_of_${MODEL}_on_${DATASET}_vcrop"