# BBO Capstone — Module 12 Submission

## Part 1 — Queries (one per function, six decimal places)

```
F1: 0.420239-0.460935
F2: 0.712703-0.491884
F3: 0.384157-0.620463-0.509763
F4: 0.390316-0.378034-0.399652-0.409505
F5: 0.390058-1.000000-1.000000-1.000000
F6: 0.370603-0.312837-0.595222-0.742241-0.115246
F7: 0.004734-0.174692-0.698721-0.351930-0.329796-0.662120
F8: 0.140922-0.100310-0.081891-0.238235-0.955016-0.263426-0.006950-0.887825
```

---

## Part 2 — Reflection (discussion board)

**How the strategy has evolved.** My first rounds were essentially generic Bayesian
optimisation: a GP with a default RBF kernel and uncertainty-driven sampling. With
only a few observations the hyperparameters collapsed to tiny length scales, producing
an overconfident, spiky surrogate that chased noise. The biggest structural shift was
taking control of the surrogate instead of trusting automatic fitting — fixing or
bounding length scales, switching to a Matérn kernel where the response looked rough,
and choosing the acquisition function deliberately (UCB while uncertainty was still
worth paying for, EI once a clear basin emerged). My loop is now systematic: append the
new observation, refit, inspect the learned length scales and the current best, then
decide between widening exploration or tightening a local exploitation box.

**Principal-component-like drivers of variation.** My clearest "principal components"
are the input dimensions the GP finds sensitive — those with short length scales under
ARD kernels. In the 8D function, dimensions 6 and 8 drove almost no variation (their
length scales ran to the upper bound, ~20–100) while the rest stayed tight; the 6D
function showed the same for one coordinate. A few axes explain most of the change in
*y*, exactly as a few principal components capture most variance. A second dominant
direction is structural: several functions are maximised by pushing variables to the
boundary (function 5 sits at x2=x3=x4=1), so proximity-to-corner is itself a major
axis of variation.

**What to keep exploring versus simplify.** I treat long length scales as low-variance
components to drop: once a dimension looks irrelevant I stop searching it and pin or
freely vary it, collapsing the effective 8D search toward 5–6D. I keep exploring the
short-length-scale axes and any region where posterior std is still large relative to
the achievable gain. The rule is a cost/benefit on uncertainty — if a dimension or
region no longer moves *y*, querying there is redundant, the BBO equivalent of carrying
a near-zero-variance component.

**Influence on the final round (Module 24).** This round mostly confirmed exploit-heavy
bets paying off — six of eight functions improved — so I'm now close to local optima in
several. For the final round I'll shift slightly back toward exploration on the
plateaued functions (2 and 3 didn't improve), while keeping tight exploitation on those
still climbing (5 is monotone toward a corner, 1 just jumped four orders of magnitude).
With the query budget nearly spent, the right move is a final verification near each
suspected optimum rather than speculative jumps.

**PCA insights applied to BBO.** The core PCA lesson — concentrate on variance, discard
redundancy — maps directly onto reading ARD length scales as a relevance ranking, and
onto avoiding near-duplicate queries (I already mask candidates within a minimum
distance, the same instinct as dropping correlated features). It also reframes the
objective: I'm not modelling all eight dimensions equally, I'm locating the
low-dimensional manifold where the variation lives and spending my limited queries there.
