## Baseline document

Parameter I'm changing are:
- the number of hypothesis tested.
- the proportions of true null hps
- mean zero levels and level values
- level distribution

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