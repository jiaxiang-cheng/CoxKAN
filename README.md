# CoxKAN: Extending Cox Proportional Hazards Model with Symbolic Non-Linear Log-Risk Functions for Survival Analysis

**Abstract**: The Cox proportional hazards (CPH)
model has been widely applied in survival analysis
to estimate relative risks across different subjects given multiple covariates.
Traditional CPH models rely on a linear combination of covariates weighted with coefficients as the log-risk function,
which imposes a strong and restrictive assumption, limiting generalization.
Recent deep learning methods enable non-linear log-risk functions.
However, they often lack interpretability due to the end-to-end training mechanisms.
The implementation of Kolmogorov-Arnold Networks (KAN)
offers new possibilities for extending the CPH model with fully transparent and symbolic non-linear log-risk functions.
In this paper, we introduce CoxKAN,
a novel model for survival analysis
that leverages KAN to enable a non-linear mapping from covariates to survival outcomes in a fully symbolic manner.
CoxKAN maintains the interpretability of traditional CPH models
while allowing for the estimation of non-linear log-risk functions.
Experiments conducted on both synthetic data and various public benchmarks demonstrate
that CoxKAN achieves competitive performance in terms of prediction accuracy and exhibits superior interpretability compared to current state-of-the-art methods.

## Reproducibility

Run the following command to reproduce all experimental results
as summarized in Table 1:
```
bash exp/test.txt
bash exp/test-linear.txt
```

Run the following command to reproduce all ablation studies
as summarized in Figure 8:
```
bash exp/test-ablation-lamb.txt
bash exp/test-ablation-order.txt
```

Check out our notebooks for automatically summarizing all
experimental results from the raw outputs of runs above:
```
notebook/Summary.ipynb
notebook/Symbolification.ipynb
```