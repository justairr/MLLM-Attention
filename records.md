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
|                               | MME          | 1666.53         |
| QwenVL2 keep30%linear[0.95~1] | InfoVQA_VAL  | 68.5368         |
|                               | ChartQA_TEST | 71.76           |
| QwenVL2 keep30%linear[0.99~1] | InfoVQA_VAL  | 68.758          |
|                               | CharQA_TEST  | 71.56           |
| QwenVL2 keep30%Nolinear[0]    | infoVQA_VAL  | 32.0689         |
|                               | DocVQA_VAL   | 22.1865         |
|                               | TextVQA_VAL  | 17.256          |
|                               | ChartQA_TEST | 38.32           |
| QwenVL2 keep30%SM[0,-3]       | MME          | 1669.19         |
|                               | infoVQA_VAL  | 67.1466         |
|                               | DocVQA_VAL   | 88.7199         |
| QwenVL2 keep100%              | infoVQA_VAL  | 68.8296/68.6629 |
|                               | ChartQA_TEST | 71.6            |
|                               | MME          | 1669.05         |
|                               |              |                 |

| 方法和参数(min256max256)      | 数据集             | 效果            |
| ----------------------------- | ------------------ | --------------- |
| QwenVL2 keep30%exp            | infoVQA_VAL        | 46.32           |
|                               | DocVQA_VAL         | 75.2352         |
|                               | TextVQA_VAL        | 70.434          |
|                               | MMMU_DEV           | 0.4             |
|                               | MMMU_VAL           | 0.4944          |
| QwenVL2 keep30%linear[0.0~1]  | infoVQA_VAL        | 46.2952         |
|                               | TextVQA_VAL        | 70.18           |
| QwenVL2 keep30%linear[0.3~1]  | infoVQA_VAL        | 47.2121         |
|                               | TextVQA_VAL        | 70.376          |
|                               | MMMU_DEV           | 0.44            |
|                               | MMMU_VAL           | 0.5044          |
| QwenVL2 keep30%linear[0.6~1]  | MMMU_DEV           | 0.46            |
|                               | MMMU_VAL           | 0.4867          |
|                               | ChartQA_TEST       | 64.6            |
|                               | MMBench_DEV_EN_V11 | 0.76238         |
|                               | InfoVQA_VAL        | 47.7767         |
| QwenVL2 keep30%linear[0.9~1]  | MMMU_DEV           | 0.42            |
|                               | MMMU_VAL           | 0.4967          |
|                               | ChartQA_TEST       | 64.72           |
| QwenVL2 keep30%linear[0.95~1] | ChartQA_TEST       | 64.76           |
| QwenVL2 keep40%linear[0.8~1]  | ChartQA_TEST       | 64.52           |
| QwenVL2 keep80%linear[0.6~1]  | ChartQA_TEST       | 64.52           |
| QwenVL2 keep80%linear[0.9~1]  | MMMU_DEV           | 0.42            |
|                               | MMMU_VAL           | 0.5022          |
| QwenVL2 keep100%              | ChartQA_TEST       | 64.8            |
|                               | DocVQA_VAL         | 75.4715         |
|                               | infoVQA_VAL        | 47.7104         |
|                               | TextVQA_VAL        | 70.534          |
|                               | MMMU_DEV           | 0.4333/0.42     |
|                               | MMMU_VAL           | 0.49111/0.50222 |
|                               | MME                | 1676.14         |
|                               | MMBench_DEV_EN_V11 | 0.76006         |
| QwenVL2 keep3 x105%           | ChartQA_TEST       | 64.88           |
| QwenVL2 keep3 x110%           | ChartQA_TEST       | 64.36           |
| QwenVL2 keep3 x115%           | ChartQA_TEST       | 64.08           |
| QwenVL2 keep120%              | infoVQA_VAL        | 48.0962         |

| 方法和参数(min256max256, optimized prompt) | 数据集      | 效果    |
| ------------------------------------------ | ----------- | ------- |
| QwenVL2 keep30%exp                         | infoVQA_VAL | 46.2995 |
|                                            |             |         |
