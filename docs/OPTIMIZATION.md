For each optimization implemented:
Profiling Evidence:
Problem identified: What bottleneck or issue did profiling reveal?
Solution implemented: What optimization strategy did you apply?
Code comparison: Show before/after code snippets (can be brief)
Performance impact: Runtime improvement, memory savings, or stability gains
Trade-offs: Any costs (code complexity, readability, maintenance, precision)?
Include profiler output or visualizations showing improvements
Show flame graphs, timing comparisons, or memory plots if helpful
Which optimizations provided the best return on investment?
What surprised you about where time was actually spent?
Which optimizations were not worth the effort?

## Optimization summary

This is a brief outline of the improvement I made to speed up the code:
1) As mentioned in `baseline.md` The profiler showed that the main bottleneck is the repeated concatenation of pandas DataFrames. This can be very easily sped up by appending elements to a list and then creating the Pandas DataFrame only when returning the results.
2) According to profiler results the second big bottleneck is the function `cdf` in `2*(1-cdf(scores))` called during p-value computation. This can be replaced with the more efficient `1-special.erfc(np.abs(z_scores) / np.sqrt(2))`. Time complexity should be the same but `special.erf` is 10x faster in practice ($\approx 1e-3$ vs $\approx 1e-4$ on $n=100$)
3) Profiler shows that the next bottleneck are the functions computing the metrics (e.g. FDR). This was already vectorised in my original code. The next improvement I can implement is to use `numba` to compile the code in C. This adds a bit of overhead at the beginning (because of compilation) but runs faster. The gains are especially relevant as `nsim` grows since the compilation costs gets amortised over a larger number of iterations.
4) Profiler shows next bottleneck is the function `generate_mean` to generate the non-zero null hypothesis. As before this is already vectorised and to make it faster I can compile it in C using numba. There are a few very small modifications required to make this run efficiently with numba such as moving the shuffling of the array outside the function and preallocating part of the memory before calling it (note: this is not strictly required, the computation of p-values should not depend on their order but it felt like the right thing to do. It also adds very little computation time). It also seems like adding the signature explicitly with `@njit(float64[:](int64, int64, int64, int64))` speeds up the code a little bit. I suspect it is due to more efficient compilation. Inded adding the signature makes numba compile the function as soon as it is imported instead of waiting for the first input, evaluating the type and then compiling. In the end it should not matter as much since compilation time is just amortised over all runs.

To see the improvement I run different versions of the simulation with sequential improvments. Note that al of these results do not take into account the time to save files. Moreover all simulations return the same results, up to numerical approximation.
On 1k simulation runs:
- Old version with parallelization takes $21.2s \pm 0.28$
- Removing pd concatenation takes $34.7 \pm 0.15$ and $19.5s\pm 0.12$ with parallelisation.
- Changing to erf takes $24.7s \pm 0.18$ and $16.9s \pm 0.63$ with parallelization
- Moving to numba for metrics takes $18.6\pm 0.17$ and $16.5\pm 0.44$. 
- Compiling `generate_means` with numba speeds this up to $16.3s \pm 0.23$ and $17.4s\pm 0.28$ with parallelisation. As expected the improvement with parallelisation slows down as the runtime of each iteration decreases, especially for $n_{sim}$ small. For instance on 2k simulations we have $27.8s$ with parallelisation and $34.1s$ without. Increasing the number of iterations also yield additional advantage by amortizing the compilation of numba functions.

Figure XX shows the improvement across versions for different number of simulations. The figure compares: 
- Old Method - Concat (COLOR): this is the version submitted for Unit 2. It uses parallelisation because it would be too slow to run otherwise.
- Old Method (COLOR): this is the version without pandas concatenation. It uses parallelisation because it would be too slow to run otherwise.
- New Method (COLOR): this is the version with all the optimizations mentioned above without parallisation
- New Method - Parallel (COLOR): this is the version with optimised code and with parallel exectution on 10 cores. 

We can see that for small values of $n_{sim}$ the overhead of parallelisation make the function run much slower than the optimised version (in COLOR). This disavantage disappear after $\approx 1k$ runs. It is interesting to see that, for $n_{sim}$ large enough, the old ineffeicient functions are faster than the optimsed version. This shows how powerful parallelisation is: even very inefficient code can be mad every fast by distributing the computation. 

Figure XX displays the plot of empirical vs theoretical complexity and the behaviou of the actual timing data. From this it appears that I have not run yet into the cealing of asymptotic behaviour of the function. Indeed the upper bound for complexity is $O(k*n_{sim})$ where $k$ is a constant depending on the number of scenario I'm running (explained more in detail in `BASELINE.md`). From the plot it is clear that the behaviour of the function of $n$ up to $50k$ is closer to $n^{0.6}$ rathen than $n$. 


## Regression test
The script `regression.py` runs validation tests to check if the different version produce the same results. The results are (almost) exactly the same across simulations. This was achieved by using numpy's `rng.spawn` to spawn random seed to use across simulations. This maintains consistency between parallel and sequential functions. 
