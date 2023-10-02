# Quantization

Quantization of PhiNet in PyTorch. 

## Modifications for quantization
The original model once quantized was not efficient. 

Modifications:
- Channels divisible by 8 in depthwise convolution;
- Removed ZeroPad2d and inserted padding within the convolution. 

Results:
```
FP32 size:
16.952 MB
INT8 size:
4.580 MB

FP32 CPU Inference Latency: 15.069 ms
FP32 GPU Inference Latency: 1.988 ms
INT8 CPU Inference Latency: 6.055 ms
```

Testing on MNIST and CIFAR10:
- Training in FP32 of the original model;
- Training in FP32 of the optimized model;
- Post Training Static Quantization of the optimized model;
- Quantization aware training of the optimized model.

MNIST Results:
```
Original FP32 model CPU Inference Latency: 5.241 ms
Original FP32 model GPU Inference Latency: 3.314 ms
Original FP32 model test accuracy: 98.80%
Original FP32 model training time: 92.071 s

Optimized FP32 model CPU Inference Latency: 3.427 ms
Optimized FP32 model GPU Inference Latency: 2.197 ms
Optimized FP32 model test accuracy: 98.71%
Optimized FP32 model training time: 67.730 s

Optimized INT8 model static quant CPU Inference Latency: 2.283 ms
Optimized INT8 model static quant test accuracy: 98.67%

Optimized INT8 model QAT CPU Inference Latency: 2.297 ms
Optimized INT8 model QAT test accuracy: 98.62%
Optimized INT8 model QAT training time: 305.282 s
```

## Performance of the Orginal Model vs Optimized Model
Compare the performance of the original model and the modified model by changing:
- Number of layers;
- $\alpha$;
- $\beta$;
- t<sub>0</sub>;

Results by changing the number of layers:
![output](https://github.com/Tremo8/Quantization/assets/102596472/6e3d34d4-8693-423f-95af-2c62323ea82e)
![output2](https://github.com/Tremo8/Quantization/assets/102596472/55df6fdc-12c3-4f95-8709-72ae2dda5db4)


## References
- **Paissan, Francesco, Alberto Ancilotto, and Elisabetta Farella (2022)**. "PhiNets: A Scalable Backbone for Low-Power AI at the Edge." *ACM Trans. Embed. Comput. Syst.*. [DOI: 10.1145/3510832](https://doi.org/10.1145/3510832)
