#!/bin/bash

# 全参数网格搜索脚本（含目录锁定）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/VLMEvalKit" || {
    echo "Error: Could not enter VLMEvalKit directory at $SCRIPT_DIR/VLMEvalKit"
    exit 1
}

# 实验参数配置
DATASETS=("GQA_TestDev_Balanced")
MODEL=Qwen2-VL-7B-Instruct

# 定义各参数搜索范围
KW_VALUES=(1.0)      # 权重系数
WT_VALUES=("exp") # 加权类型 "uniform"

# GPU监控配置
CHECK_INTERVAL=10
MEMORY_THRESHOLD=500
UTILIZATION_THRESHOLD=10
LOCK_DIR="/tmp/gpu_locks"
GPU_IDS=(0 1 2 3 4 5 6 7)    # 8卡配置

# 生成参数组合函数
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

# GPU检测函数（增加目录验证）
check_gpu() {
    local gpu_id=$1
    local lockfile="${LOCK_DIR}/gpu_${gpu_id}.lock"
    
    # 检查锁文件和目录状态
    [ -f $lockfile ] && return 1
    [ ! -d "$SCRIPT_DIR/VLMEvalKit" ] && echo "Directory missing!" && exit 1

    local memory_used=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
    local utilization=$(nvidia-smi -i $gpu_id --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    if [ $memory_used -lt $MEMORY_THRESHOLD ] && [ $utilization -lt $UTILIZATION_THRESHOLD ]; then
        touch $lockfile
        return 0
    fi
    return 1
}

COMPLETED_FILE="${LOCK_DIR}/completed.count"
echo 0 > "$COMPLETED_FILE"

# 任务执行函数修改
run_task() {
    local gpu_id=$1
    IFS=',' read -r dataset kp kw ls wt <<< "$2"
    
    (
        cd "$SCRIPT_DIR/VLMEvalKit" || exit 1
        
        echo "▶️ [$(cat $COMPLETED_FILE)/$total_tasks] Starting on GPU$gpu_id: Dataset=$dataset, KP=$kp, KW=$kw, LS=$ls, WT=$wt"
        
        CUDA_VISIBLE_DEVICES=$gpu_id \
        HF_HUB_OFFLINE=1 \
        KP=$kp KW=$kw LS=$ls WT=$wt \
        python run.py \
            --data "$dataset" \
            --model $MODEL \
            --verbose \
            --work-dir "$SCRIPT_DIR/Qwen_256_2b7b/${MODEL}_on_${dataset}_KP${kp}_KW${kw}_LS${ls}_${wt}"
        
        # 原子递增计数器
        flock -x 200
        completed=$(cat "$COMPLETED_FILE")
        echo $((completed + 1)) > "$COMPLETED_FILE"
        flock -u 200
        
        echo "✅ [$(cat $COMPLETED_FILE)/$total_tasks] Completed: Dataset=$dataset, KP=$kp, KW=$kw, LS=$ls, WT=$wt"
        rm -f "${LOCK_DIR}/gpu_${gpu_id}.lock"
    ) 200>"${LOCK_DIR}/counter.lock" &
}

while true; do
    completed=$(cat "$COMPLETED_FILE")
    [ $completed -ge $total_tasks ] && break
    
    for gpu in "${GPU_IDS[@]}"; do
        if check_gpu $gpu; then
            if [ ${#TASK_QUEUE[@]} -gt 0 ]; then
                task="${TASK_QUEUE[0]}"
                TASK_QUEUE=("${TASK_QUEUE[@]:1}")
                run_task $gpu "$task"
            fi
        fi
    done
    
    remaining=$((total_tasks - completed))
    echo "📊 Progress: $completed/$total_tasks done, $remaining remaining"
    sleep $CHECK_INTERVAL
done

# 等待所有后台任务
wait
echo "🎉 All tasks completed! Total: $total_tasks combinations"