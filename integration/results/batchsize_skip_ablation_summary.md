# Batch Size Skip Ablation Summary

Configuration:

- GPU: NVIDIA A40
- Model: LSUN Churches LDM-8
- Steps: 200 DDIM steps
- Samples: equal to batch size for each run
- Unit: ms/sample
- Conditions:
  - Full pipeline: no layers skipped
  - Skip attn: AttentionBlock skipped
  - Skip res: FusedResBlock/ResBlock skipped
  - Skip groupnorm: FusedGroupNormSiLU skipped
  - Skip all: AttentionBlock and ResBlock both skipped

## batch_size=84

| Mode | Full pipeline | Skip attn | Skip res | Skip groupnorm | Skip all |
|---|---:|---:|---:|---:|---:|
| FP32 | 774.7 | 458.4 | 355.3 | 728.8 | 40.1 |
| FP16 | 292.6 | 213.5 | 104.6 | 238.8 | 25.3 |
| INT8 baseline | 312.8 | 250.9 | 102.0 | 266.1 | 23.8 |
| INT4 baseline | 349.2 | 269.6 | 101.8 | 324.1 | 23.3 |
| INT8 MoDiff | 326.3 | 249.5 | 103.1 | 279.7 | 23.6 |
| INT4 MoDiff | 307.3 | 232.9 | 103.4 | 261.4 | 23.4 |

## batch_size=126

| Mode | Full pipeline | Skip attn | Skip res | Skip groupnorm | Skip all |
|---|---:|---:|---:|---:|---:|
| FP32 | 771.5 | 455.9 | 353.9 | 726.4 | 39.9 |
| FP16 | 288.5 | 208.6 | 100.0 | 234.0 | 47.7 |
| INT8 baseline | 307.4 | 231.7 | 100.1 | 261.2 | 22.2 |
| INT4 baseline | 331.2 | 255.6 | 100.1 | 287.8 | 22.1 |
| INT8 MoDiff | 320.0 | 245.2 | 100.1 | 274.8 | 21.4 |
| INT4 MoDiff | 301.4 | 228.1 | 100.3 | 256.5 | 22.1 |

## batch_size=168

| Mode | Full pipeline | Skip attn | Skip res | Skip groupnorm | Skip all |
|---|---:|---:|---:|---:|---:|
| FP32 | 754.3 | 441.1 | 352.5 | 711.1 | 39.2 |
| FP16 | 284.9 | 204.2 | 101.9 | 236.2 | 21.0 |
| INT8 baseline | 304.6 | 227.0 | 101.5 | 258.4 | 20.6 |
| INT4 baseline | 328.7 | 252.4 | 120.6 | 288.2 | 20.3 |
| INT8 MoDiff | 318.5 | 240.7 | 101.6 | 285.7 | 21.1 |
| INT4 MoDiff | 302.7 | 223.5 | 119.0 | 254.9 | 21.4 |

Raw outputs:

- `integration/results/batchsize_skip_ablation_bs84/`
- `integration/results/batchsize_skip_ablation_bs126/`
- `integration/results/batchsize_skip_ablation_bs168/`
