# Quantization

Quantization of PhiNet in PyTorch. 

## Modifications for quantization.
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

FP32 CPU Inference Latency: 29.540 ms
FP32 GPU Inference Latency: 3.490 ms
INT8 CPU Inference Latency: 14.413 ms
```

Testing on MNIST and CIFAR10:
- Training in FP32 of the original model;
- Training in FP32 of the optimized model;
- Post Training Static Quantization of the optimized model;
- Quantization aware training of the optimized model.

MNIST Results:
```
Orginal FP32 model CPU Inference Latency: 9.244 ms
Orginal FP32 model GPU Inference Latency: 5.900 ms
Orginal FP32 model test accuracy: 98.68%
Original FP32 model training time: 179.616 s

Optimized FP32 model CPU Inference Latency: 7.669 ms
Optimized FP32 model GPU Inference Latency: 3.823 ms
Optimized FP32 model test accuracy: 98.59%
Optimized FP32 model training time: 161.239 s

Optimized INT8 model static quant CPU Inference Latency: 3.453 ms
Optimized INT8 model static quant test accuracy: 98.36%

Optimized INT8 model QAT CPU Inference Latency: 4.760 ms
Optimized INT8 model QAT test accuracy: 98.81%
Optimized INT8 model QAT training time: 617.050 s
```

## Performance of the Orginal Model vs Optimized Model
Performance comparison of the optimized model with the original model. 

Compare the performance of the original model and the modified model by changing:
- Number of layers;
- $\alpha$;
- $\beta$;
- t<sub>0</sub>;


## References

- **Paissan, Francesco, Alberto Ancilotto, and Elisabetta Farella (2022)**. "PhiNets: A Scalable Backbone for Low-Power AI at the Edge." *ACM Trans. Embed. Comput. Syst.*. [DOI: 10.1145/3510832](https://doi.org/10.1145/3510832)
