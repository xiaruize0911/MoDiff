# AWQ Attention Layer Inventory

Mode: `int8_awq_full_baseline`

## Summary

| Owner / kernel path | Count |
|---|---:|
| AWQ GEMM | 42 |
| AWQ GEMM backend | 37 |
| MoDiff/CUTLASS INT8 Conv2d | 70 |
| PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | 21 |

## Attention Compatibility

AWQ `single_query_attention` is not used for LDM full self-attention because it is an autoregressive decode kernel for one query plus KV cache. LDM `AttentionBlock` attends all spatial tokens at once and currently uses PyTorch `scaled_dot_product_attention`.

## Layers

| Layer | Type | Owner | Shape | Role |
|---|---|---|---|---|
| `time_embed.0` | `OptimizedInt8Linear` | AWQ GEMM backend | 192->768 | linear |
| `time_embed.2` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `input_blocks.1.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.1.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->384 | linear |
| `input_blocks.1.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.1.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->576, k=1 | attention qkv/proj |
| `input_blocks.1.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.1.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->192, k=1 | attention qkv/proj |
| `input_blocks.2.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.2.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->384 | linear |
| `input_blocks.2.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.2.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->576, k=1 | attention qkv/proj |
| `input_blocks.2.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.2.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->192, k=1 | attention qkv/proj |
| `input_blocks.3.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.3.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->384 | linear |
| `input_blocks.3.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.4.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.4.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `input_blocks.4.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.4.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `input_blocks.4.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.4.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `input_blocks.5.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.5.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `input_blocks.5.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.5.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `input_blocks.5.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.5.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `input_blocks.6.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.6.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `input_blocks.6.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.7.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.7.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `input_blocks.7.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.7.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `input_blocks.7.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.7.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `input_blocks.8.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.8.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `input_blocks.8.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.8.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `input_blocks.8.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.8.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `input_blocks.9.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.9.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `input_blocks.9.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.10.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.10.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `input_blocks.10.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.10.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->2304, k=1 | attention qkv/proj |
| `input_blocks.10.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.10.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->768, k=1 | attention qkv/proj |
| `input_blocks.11.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.11.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `input_blocks.11.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.11.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->2304, k=1 | attention qkv/proj |
| `input_blocks.11.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `input_blocks.11.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->768, k=1 | attention qkv/proj |
| `input_blocks.12.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.12.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `input_blocks.12.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.13.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.13.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `input_blocks.13.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.14.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `input_blocks.14.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `input_blocks.14.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `middle_block.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `middle_block.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `middle_block.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `middle_block.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->2304, k=1 | attention qkv/proj |
| `middle_block.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `middle_block.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->768, k=1 | attention qkv/proj |
| `middle_block.2.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `middle_block.2.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `middle_block.2.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.0.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 1536->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.0.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.0.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.1.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 1536->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.1.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.1.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.2.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 1536->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.2.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.2.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.2.1.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.2.1.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.2.1.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.3.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 1536->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.3.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.3.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.3.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->2304, k=1 | attention qkv/proj |
| `output_blocks.3.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.3.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->768, k=1 | attention qkv/proj |
| `output_blocks.4.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 1536->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.4.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.4.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.4.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->2304, k=1 | attention qkv/proj |
| `output_blocks.4.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.4.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->768, k=1 | attention qkv/proj |
| `output_blocks.5.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 1152->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.5.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.5.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.5.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->2304, k=1 | attention qkv/proj |
| `output_blocks.5.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.5.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 768->768, k=1 | attention qkv/proj |
| `output_blocks.5.2.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.5.2.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->1536 | linear |
| `output_blocks.5.2.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->768, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.6.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 1152->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.6.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.6.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.6.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `output_blocks.6.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.6.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `output_blocks.7.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.7.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.7.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.7.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `output_blocks.7.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.7.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `output_blocks.8.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.8.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.8.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.8.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `output_blocks.8.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.8.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `output_blocks.8.2.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.8.2.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.8.2.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.9.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.9.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.9.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.9.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `output_blocks.9.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.9.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `output_blocks.10.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 768->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.10.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.10.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.10.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `output_blocks.10.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.10.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `output_blocks.11.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 576->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.11.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.11.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.11.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->1152, k=1 | attention qkv/proj |
| `output_blocks.11.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.11.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 384->384, k=1 | attention qkv/proj |
| `output_blocks.11.2.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.11.2.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->768 | linear |
| `output_blocks.11.2.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->384, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.12.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 576->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.12.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->384 | linear |
| `output_blocks.12.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.12.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->576, k=1 | attention qkv/proj |
| `output_blocks.12.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.12.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->192, k=1 | attention qkv/proj |
| `output_blocks.13.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.13.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->384 | linear |
| `output_blocks.13.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.13.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->576, k=1 | attention qkv/proj |
| `output_blocks.13.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.13.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->192, k=1 | attention qkv/proj |
| `output_blocks.14.0.in_layers.2` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 384->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.14.0.emb_layers.1` | `OptimizedInt8Linear` | AWQ GEMM backend | 768->384 | linear |
| `output_blocks.14.0.out_layers.3` | `OptimizedInt8Conv2d` | MoDiff/CUTLASS INT8 Conv2d | 192->192, k=(3, 3), stride=(1, 1), pad=(1, 1) | spatial conv2d |
| `output_blocks.14.1.qkv` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->576, k=1 | attention qkv/proj |
| `output_blocks.14.1.attention` | `QKVAttentionLegacy` | PyTorch scaled_dot_product_attention (not AWQ single_query_attention) | heads=8 | full-token attention core |
| `output_blocks.14.1.proj_out` | `AWQW8A8Conv1d1x1` | AWQ GEMM | 192->192, k=1 | attention qkv/proj |
