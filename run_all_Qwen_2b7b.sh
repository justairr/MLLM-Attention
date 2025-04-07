#!/bin/bash

# 双卡并行全参数网格搜索脚本
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/VLMEvalKit" || {
    echo "Error: Could not enter VLMEvalKit directory at $SCRIPT_DIR/VLMEvalKit"
    exit 1
}

# 实验参数配置
DATASETS=("MME-RealWorld-Lite")
MODEL=Qwen2-VL-7B-Instruct

# 定义各参数搜索范围
KW_VALUES=(1.0)      # 权重系数
WT_VALUES=("exp" "linear" "uniform")    # 加权类型 "uniform"

# GPU监控配置
CHECK_INTERVAL=10
MEMORY_THRESHOLD=500
UTILIZATION_THRESHOLD=10
LOCK_DIR="/tmp/gpu_locks"
GPU_IDS=(2 3 4 5 6 7)    # 8卡配置

# 生成参数组合函数（保持原样）
generate_combinations() {
    for dataset in "${DATASETS[@]}"; do
        for wt in "${WT_VALUES[@]}"; do
            if [ "$wt" == "linear" ]; then
                kp=0.4
                ls=0.6
            elif [ "$wt" == "uniform" ]; then
                kp=0.5
                ls=0.5
            elif [ "$wt" == "exp" ]; then
                kp=0.5
                ls=0.5
            fi
            for kw in "${KW_VALUES[@]}"; do
                echo "$dataset,$kp,$kw,$ls,$wt"
            done
        done
    done
}

# 创建锁目录和任务队列
mkdir -p $LOCK_DIR
TASK_QUEUE=()
while IFS= read -r line; do
    TASK_QUEUE+=("$line")
done < <(generate_combinations)
total_tasks=${#TASK_QUEUE[@]}
completed_tasks=0

# 清理函数
cleanup() {
    rm -rf $LOCK_DIR
    echo "Cleaned up lock files"
}
trap cleanup EXIT

# 双GPU检测函数（修正变量作用域问题）
check_gpu_pair() {
    # 遍历所有可能的GPU组合
    for first in "${GPU_IDS[@]}"; do
        for second in "${GPU_IDS[@]}"; do
            if [ $first -ne $second ]; then
                local lock1="${LOCK_DIR}/gpu_${first}.lock"
                local lock2="${LOCK_DIR}/gpu_${second}.lock"
                
                # 检查是否已锁定
                [ -f $lock1 ] && continue
                [ -f $lock2 ] && continue
                
                # 检查资源使用情况
                local mem1=$(nvidia-smi -i $first --query-gpu=memory.used --format=csv,noheader,nounits)
                local util1=$(nvidia-smi -i $first --query-gpu=utilization.gpu --format=csv,noheader,nounits)
                local mem2=$(nvidia-smi -i $second --query-gpu=memory.used --format=csv,noheader,nounits)
                local util2=$(nvidia-smi -i $second --query-gpu=utilization.gpu --format=csv,noheader,nounits)
                
                if [ $mem1 -lt $MEMORY_THRESHOLD ] && [ $util1 -lt $UTILIZATION_THRESHOLD ] &&
                   [ $mem2 -lt $MEMORY_THRESHOLD ] && [ $util2 -lt $UTILIZATION_THRESHOLD ]; then
                    # 锁定GPU并返回配对
                    touch $lock1 $lock2
                    echo "$first $second"
                    return 0
                fi
            fi
        done
    done
    return 1
}

COMPLETED_FILE="${LOCK_DIR}/completed.count"
echo 0 > "$COMPLETED_FILE"

# 修改后的任务执行函数（使用双GPU）
run_task() {
    local gpu1=$1
    local gpu2=$2
    IFS=',' read -r dataset kp kw ls wt <<< "$3"
    
    (
        cd "$SCRIPT_DIR/VLMEvalKit" || exit 1
        
        echo "▶️ [$(cat $COMPLETED_FILE)/$total_tasks] Starting on GPU${gpu1}+GPU${gpu2}: Dataset=$dataset, KP=$kp, KW=$kw, LS=$ls, WT=$wt"
        
        CUDA_VISIBLE_DEVICES="$gpu1,$gpu2" \
        HF_HUB_OFFLINE=1 \
        KP=$kp KW=$kw LS=$ls WT=$wt \
        python run.py \
            --data "$dataset" \
            --model $MODEL \
            --verbose \
            --reuse \
            --work-dir "$SCRIPT_DIR/Qwen_256_2b7b/${MODEL}_on_${dataset}_KP${kp}_KW${kw}_LS${ls}_${wt}"
        
        # 原子递增计数器
        flock -x 200
        completed=$(cat "$COMPLETED_FILE")
        echo $((completed + 1)) > "$COMPLETED_FILE"
        flock -u 200
        
        echo "✅ [$(cat $COMPLETED_FILE)/$total_tasks] Completed: Dataset=$dataset, KP=$kp, KW=$kw, LS=$ls, WT=$wt"
        rm -f "${LOCK_DIR}/gpu_${gpu1}.lock" "${LOCK_DIR}/gpu_${gpu2}.lock"
    ) 200>"${LOCK_DIR}/counter.lock" &
}

while true; do
    completed=$(cat "$COMPLETED_FILE")
    [ $completed -ge $total_tasks ] && break
    
    # 获取可用GPU对（修正变量传递）
    if pair=$(check_gpu_pair); then
        gpu_pair=($pair)
        if [ ${#TASK_QUEUE[@]} -gt 0 ]; then
            task="${TASK_QUEUE[0]}"
            TASK_QUEUE=("${TASK_QUEUE[@]:1}")
            run_task ${gpu_pair[0]} ${gpu_pair[1]} "$task"
        else
            # 释放未使用的GPU对
            rm -f "${LOCK_DIR}/gpu_${gpu_pair[0]}.lock" "${LOCK_DIR}/gpu_${gpu_pair[1]}.lock"
        fi
    fi
    
    remaining=$((total_tasks - completed))
    echo "📊 Progress: $completed/$total_tasks done, $remaining remaining"
    sleep $CHECK_INTERVAL
done

# 等待所有后台任务
wait
echo "🎉 All tasks completed! Total: $total_tasks combinations"