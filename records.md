default-pixels

| Qwen-original | DocVQA_VAL  | 93.868  |
| ------------- | ----------- | ------- |
|               | InfoVQA_VAL | 75.0152 |
|               |             |         |

low-pixels (768\*28*28)

| Qwen-modified | DocVQA_VAL  | 87.50 |
| ------------- | ----------- | ----- |
|               | InfoVQA_VAL | 65.24 |
|               |             |       |

min 768 max 1280

| Qwen-Origin                        | InfoVQA_VAL | 72.7621 |
| ---------------------------------- | ----------- | ------- |
|                                    | DocVQA_VAL  | 93.2298 |
|                                    | TextVQA_VAL | 78.244  |
|                                    | MMMU_DEV    | 0.4667  |
|                                    | MMMU_VAL    | 0.4867  |
| Qwen-only_set_return_attentions    | InfoVQA_VAL | 67.9175 |
| Qwen-modified-old                  | InfoVQA_VAL | 68.9353 |
|                                    | DocVQA_VAL  | 89.4575 |
| Qwen-modified-keep60%linear[0-1]   | InfoVQA_VAL | 53.5074 |
| Qwen-modified-keep80%linear[0.6-1] | DocVQA_VAL  | 77.6851 |
|                                    | InfoVQA_VAL | 54.3744 |
| Qwen-modified-keep30%linear[0.6-1] | DocVQA_VAL  | 77.7815 |
|                                    | InfoVQA_VAL | 54.3235 |
| Qwen-modified-keep30%linear[0-0]   | InfoVQA_VAL | 51.9872 |
| Qwen-modified-keep100%             | InfoVQA_VAL | 54.6141 |
| Qwen-not-modified(inputs_embed)    | InfoVQA_VAL | 53.341  |
|                                    | InfoVQA_VAL | 61.2975 |
|                                    | DocVQA_VAL  | 77.9    |
|                                    | MMMU_DEV    | 0.47    |
|                                    | MMMU_VAL    | 0.44    |

| 方法和参数                    | 数据集       | 效果            |
| ----------------------------- | ------------ | --------------- |
| QwenVL2 keep30%linear[0~1]    | ChartQA_TEST | 70.64           |
| QwenVL2 keep30%linear[0.6~1]  | ChartQA_TEST | 71.96           |
| QwenVL2 keep30%linear[0.9~1]  | DocVQA_VAL   | 89.3121         |
|                               | infoVQA_VAL  | 68.8448         |
|                               | TextVQA_VAL  | 78.286          |
|                               | ChartQA_TEST | 71.56           |
| QwenVL2 keep30%linear[0.95~1] | InfoVQA_VAL  | 68.5368         |
|                               | ChartQA_TEST | 71.76           |
| QwenVL2 keep30%linear[0.99~1] | InfoVQA_VAL  | 68.758          |
| QwenVL2 keep30%Nolinear[0]    | infoVQA_VAL  | 32.0689         |
|                               | DocVQA_VAL   | 22.1865         |
|                               | TextVQA_VAL  | 17.256          |
|                               | ChartQA_TEST | 38.32           |
| QwenVL2 keep100%              | infoVQA_VAL  | 68.8296/68.6629 |
|                               | ChartQA_TEST | 71.6            |
|                               | MME          | 1669.05         |
