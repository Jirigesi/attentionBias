# Attention Bias Analysis in Transformer based models 

## Basics

### Transformer multi-head self-attention 

<p align="center">
  <img src="https://github.com/Jirigesi/attentionBias/blob/main/imgs/transformer_multi-headed_self-attention-recap.png" width="650" title="hover text">
</p>

### Visualization of the BERT attention

<p align="center">
  <img src="https://github.com/Jirigesi/attentionBias/blob/main/imgs/Visualization-of-the-vanilla-BERT-attention-left-and-syntax-guided-self-attention.png" width="450" title="hover text">
</p>

## Attribution analysis 

The goal of attribution analysis is attributing the predition of a deep learning network to its input features. Formally, suppose we have a fuction $F$ : $R^n -> [0, 1]$ that represents a deep learning network, and an input $x = (x_1, ... , x_n) \in R^n$. An attribution of the prediction at input $x$ relative to a baseline input $x'$ is a vector $A_f(x,x') = (a_1, ... , a_n) \in R^n$ where $a_i$ is the contribution of $x_i$ to the prediction $F(x)$.
