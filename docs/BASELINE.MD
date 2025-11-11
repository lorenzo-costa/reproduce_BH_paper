## Baseline document

### Run-time profiling
The previous version took ~40 minutes on 20k simulations using parallelisation with 10 parallel processes. To get a better estimate I also run this without parallelisation but using 2k simulations which takes ~10 minutes (I know it's embarassingly slow).

The profiler shows that the bottleneck is the repeated concatenation of pd.DataFrames within the loop. This is the most obvious speed up which drives down the computational complexity by almost 50x.

According to profiler results the second big bottleneck is the function `cdf` in `2*(1-cdf(scores))` called during p-value computation. This can be replaced with the more efficient

On 2k simulation runs:
- Old version with parallelization takes ~43s on 2k
- Removing pd concatenation takes ~70s and ~35s with paralelisation
- 

### Computational complexity analysis
The parameters I am changing throughout the study are:
- the number of hypothesis tested
- the proportions of true null hps
- the levels for non-zero means null hp 
- the distribution level values (scheme type)

Breakdown of the procedure:
- the script runs $n_{sim}$ simulations
- in each of these runs we:
	- loop over the values of m and generate the samples. this happens only once per value of m and takes $O(m)$ 
	- loop over all scenarios and run it
		- in each scenario we:
			- generate the non-zero null hypothesis. 
				this takes $O(m1)$ (i.e. number of false nulls)
			- compute the p-values
				this takes $O(m)$
			- apply the methods
				this takes $O(mlogm)$ because of the sorting or $O(m)$ for Bonferroni
			- compute the metrics.
				each metric takes $O(m)$

Note that for each iteration of $n_{sim}$ we loop over all values of $m$ and for each of those we run all the scenarios. If we denote as $n_{scenario}$ the number of scenarios and $K$ the number of metrics, the computational complexity is:
$$
O\left(n_{sim} \cdot n_{scenarios} \cdot \sum m_{i}(\log m_{i}+K)\right)
$$
In particular in our case we have 4 non-zero mean fractions, 3 scheme types, 2 non-zero mean levels, 3 methods.  It means we have $n_{sceario}=4*3*2*3 = 72$ different scenarios. The actual value of these parameters does not influence computational cost, what matter is how many different values we try.
Note also that in this case the number of metrics compute is fixed at three (hence not really relevant for complexity analysis) but it is still something to consider. 