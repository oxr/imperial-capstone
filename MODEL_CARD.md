# Datasheet for BBO Capstone Project Data Set

## 1. Motivation

This data set was created for the Bayesian Black-Box Optimisation (BBO) capstone project. The task was to optimise eight unknown objective functions by submitting a limited number of query points and observing the returned function values. Since the objective functions were hidden, the data set supports surrogate modelling, sequential decision-making and exploration–exploitation analysis.

The purpose of collecting this data was not simply to store function evaluations, but to document how query choices evolved over time. The data set records both the numerical inputs and outputs used to guide the optimisation process. It therefore supports reproducibility, reflection and comparison of different BBO strategies.

## 2. Composition

The data set contains query points and observed objective values for eight unknown functions. Each function has a different input dimensionality, ranging from low-dimensional two-dimensional functions to higher-dimensional functions with up to eight input variables. Each input coordinate lies in the interval , and each query returns a scalar objective value.

The data is stored primarily as NumPy arrays, using  and  files provided for each function, with additional submitted query points manually appended in the optimisation scripts. The working data set therefore consists of:

* original initial input/output arrays;
* manually submitted query points;
* returned scalar objective values;
* comments documenting strategy changes, duplicate submissions and observed behaviour.

There are gaps in the data because the query budget is limited. Some regions of the search space are densely sampled, especially around promising local optima, while other regions remain unexplored. For functions where local exploitation became successful, the data is biased toward high-performing regions. For some functions, broad exploratory points were tested after exploitation stalled, but these did not always improve performance.

## 3. Collection Process

The initial data was provided as part of the capstone project. Additional data was collected over ten query rounds through the project portal. In each round, one query point was submitted for each function, and the returned function value was added to the local working data set.

The query-generation strategy evolved over time. Early rounds used Gaussian Process surrogate models with acquisition functions such as Expected Improvement (EI) and Upper Confidence Bound (UCB). Later rounds became more function-specific. For some functions, I used local candidate boxes around the best observed point. For others, I used maximin exploration to place a query in a large under-sampled region after local exploitation stalled. I also introduced masks to avoid repeated or near-repeated query points.

The collection process included some manual judgement. For example, when acquisition functions selected points that were already evaluated or too close to previous queries, I modified the selection strategy. In functions where a clear boundary ridge emerged, I deliberately constrained candidate generation to that boundary. In functions where exploration failed, I returned to local refinement.

One limitation of the collection process is that a few duplicate submissions occurred. For functions 6–8, repeated points were submitted accidentally in one round, producing duplicate or non-informative results. These were treated as confirmations rather than new independent observations.

## 4. Preprocessing and Uses

The main preprocessing step was appending new query points and outputs to the original arrays. In the Gaussian Process models, outputs were normalised internally using . Candidate points were generated either uniformly across the domain, uniformly inside local boxes, or by local Gaussian sampling around the current best point. No feature engineering was applied beyond these candidate-generation choices.

The intended use of the data set is to support the BBO capstone project: fitting surrogate models, selecting new query points, evaluating optimisation progress and documenting strategy. The data can also be used for educational analysis of exploration–exploitation trade-offs, acquisition functions and surrogate-model behaviour.

Inappropriate uses include treating the data as a complete representation of the unknown functions, using it to make claims about the global optimum without qualification, or assuming that the sampling distribution is unbiased. The data is highly sequential and strategy-dependent, so it should not be interpreted as an independently and uniformly sampled benchmark data set.

## 5. Distribution and Maintenance

The data set is maintained in the GitHub repository for the BBO capstone project. The repository contains the original provided data files, optimisation scripts and appended query histories. The datasheet and model card are linked from the main README file.

The data is intended for academic and educational use within the capstone project. The original function data and portal results are part of the programme activity, so reuse should respect the course rules and any terms attached to the capstone materials.

The maintainer is the project author. Maintenance consists of updating query histories, documenting strategy changes, correcting duplicate entries where appropriate and ensuring that scripts remain reproducible.

