import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List

rng = np.random.default_rng()

def parse_clause_str(clause_str: str):
  return np.array([ int(x) for x in clause_str.split()[1:-1]])

def parse_assignment_str(assignment_str: str):
  return np.array([ int(x) for x in assignment_str])

def parse_wdimacs(wdimacs):
  """
    Returns
      Tuple:
      * n (int): number of variables
      * m (int): number of clauses
      * clauses (list of lists of ints): A list of clauses representing the MAX-SAT problem.
  """

  with open(wdimacs) as f: s = f.read()

  lines = s.splitlines()
  n = None
  m = None
  clauses = []
  for line in lines:
    match line[0]:
      case 'c':
        continue
      case 'p':
        elements = line.split(' ')
        n = int(elements[2])
        m = int(elements[3])
      case _:
        clauses.append(parse_clause_str(line))
  
  return n, m, clauses


def sat_check(clause: List[int], assignment: List[int]):
  
  indicies = abs(clause)-1         # calculate the selected indicies (-1 as starts from 1)
  flips = (clause < 0).astype(int) # calculate flips (-ve numbers mean invert)

  selected_values = assignment[indicies]

  flipped_values = np.logical_xor(selected_values,flips)
  
  # print("clause_ints\t\t", clause_ints)
  # print("assignment_values\t", assignment_values)
  # print("indicies\t\t", indicies)
  # print("flips\t\t\t", flips)
  # print("selected_values\t\t", selected_values)
  # print("flipped_values\t\t", flipped_values)

  return 1 if flipped_values.any() else 0


def n_sat(clauses: List[List[int]], assignment: List[int]):
  
  satisfied = np.array([sat_check(clause, assignment) for clause in clauses])

  return satisfied.sum()

# def population_n_sat(clauses: List[np.ndarray], population: np.ndarray):
    
#     pop_size = population.shape[0]
    
#     fitness = np.zeros(pop_size, dtype=int)
#     for clause in clauses:
#       # same as sat_check method but with entire population

#       indices = np.abs(clause) - 1
#       flips = (clause < 0).astype(int)
      
#       selected_values = population[:, indices]
      
#       flipped_values = np.logical_xor(selected_values, flips)
      
#       # Check if any literal is true in each individual: (pop_size,)
#       clause_satisfied = np.any(flipped_values, axis=1).astype(int)
      
#       fitness += clause_satisfied
    
#     return fitness

def population_n_sat(clauses, population):
    from collections import defaultdict
    pop_size = population.shape[0]
    fitness = np.zeros(pop_size, dtype=np.int32)
    
    clauses_by_len = defaultdict(list)
    for clause in clauses:
        clauses_by_len[len(clause)].append(clause)
    
    for clause_len, clause_group in clauses_by_len.items():
        if not clause_group:
            continue
        
        clause_arr  = np.array(clause_group, dtype=np.int32)

        indices     = np.abs(clause_arr) - 1
        flips       = (clause_arr < 0).astype(int)

        selected_values  = population[:, indices]
        flipped_values   = np.logical_xor(selected_values, flips)
        clause_satisfied = np.any(flipped_values, axis=2)
        
        fitness += np.sum(clause_satisfied, axis=1)
    
    return fitness

def evolutionary_algorithm(n: int, m: int, clauses: List[List[int]], time_budget: int, **kwargs):
  """
  Solves a maximum satisfiability (MAX-SAT) problem using an evolutionary algorithm.
  
  Args:
    n (int): number of variables
    m (int): number of clauses
    clauses (list): A list of clauses representing the MAX-SAT problem.
      Each clause is a logical expression that can be satisfied or unsatisfied.
    time_budget (float): The maximum amount of time (in seconds) allowed for the algorithm to run.
  
  Returns:
    triple of [t, nsat, xbest], where
      * t     is the runtime (number of generations * population size)
      * nsat  is the number of satisfies clauses in the returned solution
      * xbest is the solution found by the algorithm
  
  """

  ax      = kwargs.get('ax')       or False
  line    = kwargs.get('line')     or False
  verbose = kwargs.get('verbose')  or False

  # model arguments
  population_size = int(kwargs.get('population_size')  or 100)
  crossover_prob  = float(kwargs.get('crossover_prob') or 0.85)
  mutation_prob   = float(kwargs.get('mutation_prob')  or 3 / n)

  parents_ratio   = float(kwargs.get('parents_ratio')   or 0.2)
  offspring_ratio = float(kwargs.get('offspring_ratio') or 0.8)

  parents_size    = int(population_size * parents_ratio)
  offspring_size  = int(population_size * offspring_ratio)

  if verbose:
    print({
      'population_size': population_size,
      'crossover_prob': crossover_prob,
      'mutation_prob': mutation_prob,
      'parents_size': parents_size,
      'offspring_size': offspring_size,
    })

  start = time.time()
  end   = start + time_budget

  # initialise population
  population      = rng.integers(2, size=(population_size, n))
  costs           = population_n_sat(clauses, population) # np.apply_along_axis(lambda ind: n_sat(clauses, ind), 1, population)

  xbest = None
  nsat  = -1

  generation = 0
  history_generations = []
  history_best_nsat   = []
  while time.time() <= end:

    start = time.perf_counter()

    generation += 1

    parent_indexes  = np.argsort(costs)[-parents_size:]
    parents         = population[parent_indexes]

    # create offspring_size offspring from parents
    offspring  = np.empty((offspring_size, n), dtype=population.dtype)
    sampled    = rng.integers(0, parents_size, size=offspring_size)
    offspring  = parents[sampled].copy()

    # crossover
    for i in range(0, offspring_size):
      if rng.random() < crossover_prob:
        o1,o2 = np.random.choice(len(offspring), size=2)
        crossover = rng.integers(0, n)
        c1 = np.concatenate((offspring[o1][0:crossover], offspring[o2][crossover:n]))
        c2 = np.concatenate((offspring[o2][0:crossover], offspring[o1][crossover:n]))
        offspring[o1] = c1
        offspring[o2] = c2
    t3 = time.perf_counter()
    
    # mutation
    for idx,child in enumerate(offspring):
      mask = rng.random(n) < mutation_prob
      offspring[idx] = np.logical_xor(offspring[idx], mask).astype(int)
    
    # truncation selection
    t4 = time.perf_counter()
    offspring_costs = population_n_sat(clauses, offspring)
    new_population  = np.concatenate((population, offspring))
    new_costs       = np.concatenate((costs, offspring_costs)) # np.apply_along_axis(lambda ind: n_sat(clauses, ind), 1, new_population)
    idxs            = np.argsort(new_costs)[-population_size:]
    t5 = time.perf_counter()

    # update population & costs (should be in order)
    population = new_population[idxs]
    costs      = new_costs[idxs]

    # Update best idx
    best_idx  = np.argmax(costs)
    best_nsat = costs[best_idx]
    best_ind  = population[best_idx]
    if (best_nsat > nsat):
      xbest = best_ind
      nsat = best_nsat
      if verbose:
        print(f"[Generation {generation}] New best: {nsat} / {m} ({(100 * nsat / m):.2f}%)")
    
    if best_nsat == m:
      # we've covered every row, we've found an optimal solution
      # so we can exit early
      break
    
    t6 = time.perf_counter()

    history_generations.append(generation)
    history_best_nsat.append(best_nsat)

    if ax and line and generation % 10 == 0:
      # update graph every 10 iterations
      line.set_xdata(history_generations)
      line.set_ydata(history_best_nsat)

      ax.relim()
      ax.autoscale_view()
      plt.draw()
      plt.pause(0.001)

    t7 = time.perf_counter()

    # print(f"Part 1: {t1 - start:.6f}s")
    # print(f"Part 2: {t2 - t1:.6f}s")
    # print(f"Part 3: {t3 - t2:.6f}s")
    # print(f"Part 4: {t4 - t3:.6f}s")
    # print(f"Part 5: {t5 - t4:.6f}s")
    # print(f"Part 6: {t6 - t5:.6f}s")
    # print(f"Part 7: {t7 - t6:.6f}s")
  
  if ax:
    # draw "crosshair" on final point
    ax.axhline(y=nsat,       linestyle=":", linewidth=1, color="#000000", alpha=0.2)
    ax.axvline(x=generation, linestyle=":", linewidth=1, color="#000000", alpha=0.2)

  t = generation * population_size
  return t,nsat,xbest