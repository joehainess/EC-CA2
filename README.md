# Evolutionary Computation CA2 - MAX-SAT Optimization

Genetic algorithm implementations for solving the Maximum Satisfiability (MAX-SAT) problem, with parameter sensitivity analysis across multiple problem instances.

## Entry Points

- **`main.py`** - Entrypoint CLI for exercises 1-3
- **`exercise5.py`** - Entrypoint for parameter modulation experiments for exercise 5
- **`parameter_sweep.py`** - Entrypoint for plotting the convergence of multiple parameter values on the same plot (extra experiment for exercise 5)

## Core Implementation

- **`src/common/maxsat.py`** - WCNF parser and μ,λ genetic algorithm
- **`src/questions/`** - Entrypoints for exercises 1-3

## Benchmark Instances

- **`benchmark_instances/`** - The selected MAX-SAT benchmark instances (`.wcnf` format)

## Visualization

- **`plots/`** - Generated plots for exercise 5

## Documentation & Deployment

- **`Dockerfile`** - Submission Docker image definition
- **`exercise.tex` / `exercise.pdf`** - Report for Exercises 4 and 5
- **`zip-assignment`** - Script to auto-generate the assignment `.zip` file
- **`ec2025cw2-jfh245.zip`** - Latest submission-ready `.zip` file

## Author

Joseph Haines (Student ID: 2327945)
