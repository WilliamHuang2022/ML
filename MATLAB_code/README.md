# ML-code
matlab code for machine learning
---
**CRFS ($l_{2, 1}$ Regularized correntropy for robust feature selection)**
$$\min_{{U}}{1-\sum_{k=1}^{n} \exp (-\frac{||({X}^{T} {U}-{Y})^k||^2_2}{\sigma^2})+||{U}||_{2,1}}$$

---
**LSDA (Locality Sensitive Discriminant Analysis)**


$$\min \sum_{i j}\left(y_{i}-y_{j}\right)^{2} W_{w, i j}$$

$$\max \sum_{i j}\left(y_{i}-y_{j}\right)^{2} W_{b, i j}$$

---
**MMC (Efficient and Robust Feature Extraction by Maximum Margin Criterion)**

$$\max \sum_{k=1}^{d} {w}^T_k\left(\mathbf{S}_{b}-\mathbf{S}_{w}\right) w_k$$

---
**RFS (Efficient and robust feature selection via joint $ℓ_{2, 1}$-norms minimization)**

$$\min_{W} \frac{1}{\gamma}||X^{T} W-Y||_{2,1}+||{W}||_{2,1}$$

---
**RSLDA (Robust Sparse Linear Discriminant Analysis)**

$$\min_{P, Q, E} \operatorname{Tr}\left(Q^{T}\left(S_{w}-u S_{b}\right) Q\right)+\lambda_{1}\||Q\||_{2,1}+\lambda_{2}\||E\||_{1}$$

$$\text { s.t. } X=P Q^{T} X+E, \quad P^{T} P=I$$
