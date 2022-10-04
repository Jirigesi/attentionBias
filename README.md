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

### Primarily attribution methods

| **Method names**     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                               |   |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| Integrated Gradients | Integrated gradients represents the integral of gradients with respect to inputs along the path from a given baseline to input. The integral can be approximated using a Riemann Sum or Gauss Legendre quadrature rule.                                                                                                                                                                                                                                                   |   |
| Gradient SHAP        | Gradient SHAP is a gradient method to compute SHAP values, which are based on Shapley values proposed in cooperative game theory. Gradient SHAP adds Gaussian noise to each input sample multiple times, selects a random point along the path between baseline and input, and computes the gradient of outputs with respect to those selected random points. The final SHAP values represent the expected value of gradients * (inputs - baselines).                     |   |
| DeepLIFT             | DeepLIFT is a back-propagation based approach that attributes a change to inputs based on the differences between the inputs and corresponding references (or baselines) for non-linear activations. As such, DeepLIFT seeks to explain the difference in the output from reference in terms of the difference in inputs from reference. DeepLIFT uses the concept of multipliers to "blame" specific neurons for the difference in output.                               |   |
| DeepLIFT SHAP        | DeepLIFT SHAP is a method extending DeepLIFT to approximate SHAP values, which are based on Shapley values proposed in cooperative game theory. DeepLIFT SHAP takes a distribution of baselines and computes the DeepLIFT attribution for each input-baseline pair and averages the resulting attributions per input example.                                                                                                                                             |   |
| Saliency             | Saliency is a simple approach for computing input attribution, returning the gradient of the output with respect to the input. This approach can be understood as taking a first-order Taylor expansion of the network at the input, and the gradients are simply the coefficients of each feature in the linear representation of the model. The absolute value of these coefficients can be taken to represent feature importance.                                      |   |
| Input X gradient     | Input X Gradient is an extension of the saliency approach, taking the gradients of the output with respect to the input and multiplying by the input feature values. One intuition for this approach considers a linear model; the gradients are simply the coefficients of each input, and the product of the input with a coefficient corresponds to the total contribution of the feature to the linear model's output.                                                |   |
| Guided GradCAM       | Guided GradCAM computes the element-wise product of guided backpropagation attributions with upsampled (layer) GradCAM attributions. GradCAM attributions are computed with respect to a given layer, and attributions are upsampled to match the input size. This approach is designed for convolutional neural networks. The chosen layer is often the last convolutional layer in the network, but any layer that is spatially aligned with the input can be provided. |   |
| Lime                 | Lime is an interpretability method that trains an interpretable surrogate model by sampling data points around a specified input example and using model evaluations at these points to train a simpler interpretable 'surrogate' model, such as a linear model.                                                                                                                                                                                                          |   |
| Kernel SAP           | Kernel SHAP is a method that uses the LIME framework to compute Shapley Values. Setting the loss function, weighting kernel and regularization terms appropriately in the LIME framework allows theoretically obtaining Shapley Values more efficiently than directly computing Shapley Values.                                                                                                                                                                           |   |
