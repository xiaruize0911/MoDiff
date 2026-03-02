# Arithmetic Intensity: Linear Layers

FLOPs counted as multiply-add × 2.  Byte counts: FP32=4B/elem; INT8=1B activation+weight, 4B output; INT4=0.5B activation+weight, 4B output.


| Shape | Count | FLOPs | FP32 Bytes | INT8 Bytes | INT4 Bytes | FP32 AI | INT8 AI | INT4 AI | INT8/FP32 | INT4/FP32 |
|-------|-------|-------|-----------|-----------|-----------|---------|---------|---------|-----------|-----------|
| in=768,out=1536,count=15 | 15 | 0.02G | 4.79MB | 1.23MB | 0.64MB | 3.94 | 15.28 | 29.40 | 3.88× | 7.46× |
| in=768,out=768,count=15 | 15 | 0.01G | 2.41MB | 0.62MB | 0.32MB | 3.92 | 15.21 | 29.26 | 3.88× | 7.47× |
| in=768,out=384,count=6 | 6 | 0.00G | 1.22MB | 0.31MB | 0.16MB | 3.88 | 15.06 | 28.98 | 3.88× | 7.47× |
| in=192,out=768,count=1 | 1 | 0.00G | 0.62MB | 0.17MB | 0.10MB | 3.80 | 13.59 | 23.81 | 3.58× | 6.26× |
