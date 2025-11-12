## Baseline document

### Run-time profiling
The previous version took ~40 minutes on 20k simulations using parallelisation with 10 parallel processes. To get a better estimate I also run this without parallelisation but using 2k simulations which takes ~10 minutes (I know it's embarassingly slow).
Note, for project 2 I just naively implemented parallelisation without profiling because the simulation was taking hours to run. This resulted in a faster but still suboptimal code. I will now try to make it better.

1) The profiler shows that the bottleneck is the repeated concatenation of pd. In my simulation loop I am extending the results dataset at each iteration using pd.concat. This clearly is super inefficient. Interstingly, the speed up I saw at the beginning using parallelisation came mostly from batching these concats rather than actual parallel computation. 
This is the most obvious speed up which drives down the computational complexity by almost 50x.
2) According to profiler results the second big bottleneck is the function `cdf` in `2*(1-cdf(scores))` called during p-value computation. This can be replaced with the more efficient `special.erfc(np.abs(z_scores) / np.sqrt(2))`
3) Profiler shows that the next bottleneck are the functions computing the metrics (e.g. FDR). This is already vectorised, the next thing I can do it to use `numba` to compile the code in C. This adds a bit of overhead at the beginning (because of compilation) but runs faster (especially as `nsim` grows)
4) Profiler shows next bottleneck is the function `generate_mean` to generate the non-zero null hypothesis. As before this is already vectorised and to make it faster I can compile it in C using numba. This can be improve by pre-allocating memory 

On 1k simulation runs (note all of these return the same results up to numerical approximation):
- Old version with parallelization takes $18.9s \pm 0.28$
- Removing pd concatenation takes $34.7 \pm 0.15$ and $17.9s\pm 0.12$ 
- Changing to erf takes $24.7s \pm 0.18$ and $16.9s \pm 0.63$ with parallelization
- Moving to numba for metrics takes $18.6\pm 0.17$ and $16.5\pm 0.44$. 
- Compiling generate means with numba speeds this up to $16.3s \pm 0.23$ and $17.4s\pm 0.28$ with parallelisation. As expected the improvement with parallelisation slows down as the runtime of each iteration decreases, especially for $n_{sim}$ small. For instance on 2k simulations we have $27.8s$ with parallelisation and $34.1s$ without.

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