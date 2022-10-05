# Attention Bias Analysis in Transformer based models 
In the past few years, transformer-based models were mainly used for code-related tasks and achieved state-of-the-art performance on most tasks. Researchers have been trying to explain transformer-based models using self-attention maps. However, it has been shown that ordinary changes in self-attention maps cannot provide useful information. Attribution analysis is a study that shows promising results in attributing the predictions of a transformer-based model to its inputs. Therefore, in this study, we empirically evaluate whether attribution analysis can better analyze the model. If yes, can we explain any bias from the attribution analysis?

## Backgrounds

### Transformer multi-head self-attention
A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and other fields. 

<p align="center">
  <img src="https://github.com/Jirigesi/attentionBias/blob/main/imgs/transformer_multi-headed_self-attention-recap.png" width="650" title="hover text">
</p>

### Visualization of the BERT attention

<p align="center">
  <img src="https://github.com/Jirigesi/attentionBias/blob/main/imgs/Visualization-of-the-vanilla-BERT-attention-left-and-syntax-guided-self-attention.png" width="450" title="hover text">
</p>

## Attribution analysis 

The goal of attribution analysis is attributing the predition of a deep learning network to its input features. Formally, suppose we have a fuction $F$ : $R^n -> [0, 1]$ that represents a deep learning network, and an input $x = (x_1, ... , x_n) \in R^n$. An attribution of the prediction at input $x$ relative to a baseline input $x'$ is a vector $A_f(x,x') = (a_1, ... , a_n) \in R^n$ where $a_i$ is the contribution of $x_i$ to the prediction $F(x)$.

### Primarily Attribution Techniques

| **Method names**     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Original paper  |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| Integrated Gradients | Integrated gradients represents the integral of gradients with respect to inputs along the path from a given baseline to input. The integral can be approximated using a Riemann Sum or Gauss Legendre quadrature rule.                                                                                                                                                                                                                                                   |  [Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365) |
| Gradient SHAP        | Gradient SHAP is a gradient method to compute SHAP values, which are based on Shapley values proposed in cooperative game theory. Gradient SHAP adds Gaussian noise to each input sample multiple times, selects a random point along the path between baseline and input, and computes the gradient of outputs with respect to those selected random points. The final SHAP values represent the expected value of gradients * (inputs - baselines).                     | [Scott et al. 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)  |
| DeepLIFT             | DeepLIFT is a back-propagation based approach that attributes a change to inputs based on the differences between the inputs and corresponding references (or baselines) for non-linear activations. As such, DeepLIFT seeks to explain the difference in the output from reference in terms of the difference in inputs from reference. DeepLIFT uses the concept of multipliers to "blame" specific neurons for the difference in output.                               | [Avanti et al. 2017](https://arxiv.org/abs/1704.02685)  |
| DeepLIFT SHAP        | DeepLIFT SHAP is a method extending DeepLIFT to approximate SHAP values, which are based on Shapley values proposed in cooperative game theory. DeepLIFT SHAP takes a distribution of baselines and computes the DeepLIFT attribution for each input-baseline pair and averages the resulting attributions per input example.                                                                                                                                             |  [Scott et al. 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) |
| Saliency             | Saliency is a simple approach for computing input attribution, returning the gradient of the output with respect to the input. This approach can be understood as taking a first-order Taylor expansion of the network at the input, and the gradients are simply the coefficients of each feature in the linear representation of the model. The absolute value of these coefficients can be taken to represent feature importance.                                      |  [Karen et al. 2013](https://arxiv.org/abs/1312.6034)|
