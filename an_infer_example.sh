cd VLMEvalKit

LS=0.9
KP=0.3
IL=False
DATASET=InfoVQA_VAL
MODEL=Qwen2-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=0,1 \
HF_HUB_OFFLINE=1 \
AUTO_SPLIT=1 \
LS=${LS} \
KP=${KP} \
IL=${IL} \
python run.py \
--data ${DATASET} \
--model ${MODEL} \
--verbose \
--work-dir "../results_of_${MODEL}_on_${DATASET}_LS${LS}_KP${KP}_IL${IL}"