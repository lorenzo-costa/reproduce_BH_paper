# Makefile

# Variables
PYTHON=python
SRC=src
DATA=data
RESULTS=results

# Default target
.PHONY: all simulate figures clean

all: simulate analyse figures

# run simulations sequential
simulate:
	$(PYTHON) -m $(SRC).run_simulation

# run simulations parallel
parallel:
	$(PYTHON) -m $(SRC).run_simulation --parallel true

# analyze results
analyse:
	$(PYTHON) -m $(SRC).analyse_data

# Generate figures
figures:
	$(PYTHON) -m $(SRC).make_plots
	@echo "Figures should now be in $(RESULTS)/figures"

# clean up caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

# run tests
test:
	pytest

# profiling

# complexity
complexity:
	$(PYTHON) comparison.py --target complexity
	$(PYTHON) -m $(SRC).make_plots --target complexity
	@echo "Complexity plot should now be in $(RESULTS)/figures"

# benchmark
benchmark:
	$(PYTHON) comparison.py --target benchmark
	$(PYTHON) -m $(SRC).make_plots --target benchmark
	@echo "Benchmark plot should now be in $(RESULTS)/figures"
