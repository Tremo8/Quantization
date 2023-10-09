# Quantization

This repository focuses on the quantization of efficient neural network architectures using PyTorch. Quantization is a technique that allows us to reduce the memory and computational requirements of deep learning models, making them more efficient for deployment on various hardware platforms.

In this repository, we primarily target two popular architectures for quantization:

- MobileNetV1
- PhiNet

## MobileNetV1
### Reference
- **Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig (2017)**. "Mobilenets: Efficient convolutional neural networks for mobile vision applications.". [arXiv preprint arXiv:1704.04861](https://arxiv.org/pdf/1704.04861.pdf%EF%BC%89)

## MobileNetV2
### Reference
- **Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh (2018)**. "Mobilenetv2: Inverted residuals and linear bottlenecks.". *Proceedings of the IEEE conference on computer vision and pattern recognition*.

## PhiNet
The original model, when quantized, was not efficient.
- Improvement to make the model quantizable;
- Reduction of inference time on GPU and CPU;

### Reference
- **Paissan, Francesco, Alberto Ancilotto, and Elisabetta Farella (2022)**. "PhiNets: A Scalable Backbone for Low-Power AI at the Edge." *ACM Trans. Embed. Comput. Syst.*. [DOI: 10.1145/3510832](https://doi.org/10.1145/3510832)
