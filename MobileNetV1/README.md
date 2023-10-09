# Quantization

Quantization of MobileNetV1 in PyTorch. 

Results:
```
FP32 size:
16.276 MB
INT8 size:
4.292 MB

FP32 model CPU Inference Latency: 7.140 ms
INT8 model static quant CPU Inference Latency: 2.062 ms
```

Testing on CIFAR10:
- Training the model in FP32;
- Post Training Static Quantization of the model;

Results:
```
FP32 model CPU Inference Latency: 7.140 ms
FP32 model GPU Inference Latency: 1.686 ms
FP32 model test accuracy: 91.92%

INT8 model static quant CPU Inference Latency: 2.062 ms
INT8 model static quant test accuracy: 88.82%
```