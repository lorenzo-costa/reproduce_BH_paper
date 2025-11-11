## Analysis

#### Reproduction of Original Results
The main simulation results from the original paper were reproduced almost exactly. The only difference I found was in the implementation of scenario D, particularly for the case $m=4$. The paper describes  setup "D" as "linearly decreasing (D) number of hypotheses away from 0 in each group;". My initial interpretation was to implement this such that the non-null means would be distributed starting from the lowest level $L/4$ (e.g., if there were only two non-nulls, I would place them at $L/4$ and $L/2$). However, implementing the code this way resulted in uniformly low power across all methods, inconsistent with the paper, which shows very high power for $m=4$.

When I instead implemented the design such that the non-nulls were distributed following the correct design scheme (equal/increasing/decreasing) but starting from the highest level (i.e., if there were only two non-nulls, they were placed at $L$ and $3L/4$ irrespective of whether the scheme was I, E or D), I obtained results that closely matched those in the paper. This discrepancy is particularly relevant for small values of $m$, as it produces significantly different results only when the number of non-zero means is smaller than the number of groups (4 in this case).

#### Evaluation of Simulation Neutrality
Overall, the simulation design appears fair. Each method was tested under identical conditions, and the data generation processes were transparent and replicable. There is, naturally, an inherent bias favoring FDR-controlling methods, as they operate under less stringent error-control criteria than FWER-controlling procedures and therefore exhibit higher power. This is however precisely the paper's stated objective: to demonstrate the power advantage of FDR control, rather than to create a perfectly neutral comparison.

The simulation settings might be somewhat idealized, particularly in their use of strong signal strengths and well-structured groupings of hypotheses. While this could limit the realism of the conclusions, it does not undermine the main insight of the study: the FDR-controlling procedures achieve higher power in structured testing scenarios.

#### Suggested Design Changes
One interesting improvement would be to compare not only the difference in power but also the corresponding differences in FDR and FWER between the two methods. While these procedures are designed to control distinct quantities, it would be informative to examine how much higher the FWER becomes when using FDR-controlling methods, illustrating the price paid for greater power. Conversely, Bonferroni-type procedures explicitly control the FWER and only indirectly the FDR; assessing how much lower than the nominal level (α) their actual FDR is could help clarify where the loss of power originates.

Another worthwhile enhancement would be to run simulations with a much larger number of hypotheses (e.g., >1000) and very few non-nulls, to assess how pronounced the power gap between FDR- and FWER-controlling methods becomes in low-signal scenarios.

#### Visualization and Insights
I focused on recreating the visualizations from the paper. The reproduced plots confirmed the main findings, with power curves and error rates closely matching the originals.

Additionally, I created a visualization showing the differences in FDR across the various methods and scenarios. This clearly highlights where the gain in power comes from: Bonferroni-type methods control the FDR much more strictly than the Benjamini–Hochberg method, with FDR values approaching zero as the number of hypotheses (m) increases.

#### Surprising or Unexpected Results
The only unexpected outcome was the sensitivity of the results to the placement of the non-null hypotheses in scenario D. A seemingly minor change in their distribution had a substantial impact on observed power. This was surprising and took some time to identify. It also highlighted how delicate simulation design can be—even a subtle implementation detail can lead to dramatically different outcomes.