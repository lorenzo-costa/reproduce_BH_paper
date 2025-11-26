# Makefile

# Variables
PYTHON=python
SRC=src
DATA=data
RESULTS=results

# Default target
.PHONY: all simulate figures clean

all: simulate analyse figures

parallel: simulate_parallel analyse figures

runtime_analysis: profile complexity benchmark single_simulation regression

# run simulations sequential
simulate:
	$(PYTHON) -m $(SRC).run_simulation

# run simulations parallel
simulate_parallel:
	$(PYTHON) -m $(SRC).run_simulation --parallel 0

# analyze results
analyse:
	$(PYTHON) -m $(SRC).analyse_data

# Generate figures
figures:
	$(PYTHON) -m $(SRC).make_plots --target plots
	@echo "Figures should now be in $(RESULTS)/figures"

# clean up caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

# run tests
test:
	pytest

# profiling
profile:
	$(PYTHON) -m $(SRC).profiling --nsim 500 --parallel 0
	@echo "Profiling results should now be in $(RESULTS)/profiling"

# complexity
complexity:
	$(PYTHON) -m $(SRC).comparison --target complexity
	$(PYTHON) -m $(SRC).make_plots --target complexity
	@echo "Complexity plot should now be in $(RESULTS)/figures"

single_simulation:
	$(PYTHON) -m $(SRC).comparison --target single_simulation
	$(PYTHON) -m $(SRC).make_plots --target single_simulation
	@echo "Single simulation complexity completed. Results should be in $(RESULTS)/figures"

# benchmark
benchmark:
	$(PYTHON) -m $(SRC).comparison --target benchmark
	$(PYTHON) -m $(SRC).make_plots --target benchmark
	@echo "Benchmark plot should now be in $(RESULTS)/figures"

regression:
	$(PYTHON) -m $(SRC).test_regression

