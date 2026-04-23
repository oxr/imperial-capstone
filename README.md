# Bayesian Black-Box Optimisation (BBO) Capstone Project

## 1. Project Overview

This project explores Bayesian Black-Box Optimisation (BBO) in the context of unknown objective functions that are expensive or impossible to evaluate analytically. The goal is to iteratively propose query points that efficiently maximise an unknown function using only observed input–output pairs.

BBO is highly relevant in real-world machine learning and engineering tasks where evaluating the objective is costly, such as hyperparameter tuning, experimental design, or financial optimisation. Instead of exhaustively searching the space, the approach builds a probabilistic surrogate model of the function and uses it to guide future queries.

This capstone project supports my broader interest in combining theoretical computer science (modelling, structure, inference) with practical ML workflows. It demonstrates the ability to reason under uncertainty, manage sparse data, and design adaptive optimisation strategies.

---

## 2. Inputs and Outputs

The optimisation process operates on black-box functions with varying dimensionality (from 2D up to 8D in this project).

**Input:**  
A query point  
x = (x1, x2, ..., xn)  
where each xi ∈ [0,1], formatted as:  

0.123456-0.654321-...

**Output:**  
A scalar function value y = f(x), returned by the system after submission.

**Example:**
Input:  0.252147-0.204355-0.175316-0.048325-0.899664-0.150554-0.281117-0.601399  
Output: 9.7940105951019

The optimiser maintains a dataset of observed (X, y) pairs and uses these to guide future queries.

---

## 3. Challenge Objectives

The objective is to maximise the unknown function value under a limited query budget.

Key constraints:
- The function is unknown and non-analytic
- Evaluations are expensive and limited
- The structure (smoothness, noise, dimensional relevance) is not given
- Query decisions must be made sequentially

The challenge is to efficiently balance:
- Exploration: sampling uncertain or unvisited regions  
- Exploitation: refining around known high-value areas  

Success is measured by how effectively the optimiser improves the best observed value within the available number of queries.

---

## 4. Technical Approach

My approach is based on Gaussian Process (GP) surrogate modelling combined with acquisition functions such as Expected Improvement (EI) and Upper Confidence Bound (UCB).

### Core Method

- Model the unknown function using a Gaussian Process
- Use kernels such as:
  - Radial Basis Function (RBF)
  - Matérn (for less smooth assumptions)
- Select query points via acquisition functions:
  - Expected Improvement (EI)
  - Upper Confidence Bound (UCB)

---

### Evolution of Strategy

**Initial phase (low data):**  
I began with a standard GP-based optimisation loop, focusing on uncertainty-driven exploration. Early results suggested relatively flat response surfaces, so the strategy emphasised cautious probing of uncertain regions rather than aggressive exploitation.

**Model-aware phase:**  
In early iterations, unconstrained hyperparameter fitting produced very small length scales, leading to overly local models. I addressed this by imposing bounds and switching to a Matérn kernel, encouraging smoother and more globally consistent behaviour. This marked a shift toward actively interpreting and managing the surrogate model.

**Exploitation phase (mid-dimensional problems):**  
Once meaningful improvements were found, the strategy became more exploitative. Expected Improvement proved effective in refining around promising regions, while UCB was used when uncertainty remained large and potential gains justified broader exploration.

**High-dimensional phase:**  
In higher dimensions, I used anisotropic (ARD) kernels to capture differences in variable importance. Some dimensions exhibited very large length scales, suggesting low relevance, while others remained sensitive. The optimisation increasingly focused on exploiting this structure, effectively reducing the problem to a lower-dimensional search.

---

### Key Insights

- Kernel hyperparameters (especially length scales) provide important diagnostic information  
- Early overfitting must be controlled to preserve meaningful exploration  
- Acquisition functions should be adapted to the scale and uncertainty of the problem  
- In high-dimensional spaces, identifying irrelevant variables is crucial  

Overall, the strategy evolved from a generic Bayesian optimisation loop into a model-aware, adaptive process that interprets and adjusts the surrogate model throughout optimisation.