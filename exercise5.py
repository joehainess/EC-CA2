import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import time
import numpy as np
import re
from src.common.maxsat import parse_wdimacs,evolutionary_algorithm
import matplotlib.pyplot as plt

def run_parallel(fn, n, args, kwargs, max_workers):
    """
    Run fn(*args, **kwargs) n times in parallel.
    Returns a list of n results in completion order.
    """
    results = [None] * n

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, *args, **kwargs): i for i in range(n)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()

    return results

if __name__ == '__main__':

  # Enable interactive mode for live updating plots
  plt.ion()

  prog_name = sys.argv[0]

  parser = argparse.ArgumentParser(prog=prog_name)
  parser.add_argument('-time_budget', required=True)
  parser.add_argument('-repetitions', required=True)
  # parser.add_argument('-wdimacs_instances', required=True)
  # parser.add_argument('-parameter',         required=True)
  # parser.add_argument('-parameter_values',  required=True)

  parser.add_argument('-max_generations')
  parser.add_argument('-verbose','-v', action='count')

  args = parser.parse_args()

  time_budget, repetitions = (vars(args)[k] for k in ['time_budget', 'repetitions'])
  kwargs = {k: v for k, v in vars(args).items() if k not in ('time_budget', 'repetitions')}

  verbose = kwargs.get('verbose') or 0
        
  time_budget = int(time_budget)
  repetitions = int(repetitions)

  instance_folder   = 'benchmark_instances'
  output_folder     = 'plots/exercise5'
  wdimacs_instances = [
    'bcp-fir/normalized-fir06_area_delay.wcnf',
    'reversi/rev66-4.wcnf',
    'pbo-mqc-nencdr/10tree305p.wcnf',
  ]

  # parameter         = 'lambda_mu_ratio'
  # parameter_values  = [2,3,4,5,6,7,8,9,10]
  parameter         = 'crossover_prob'
  parameter_values  = [0,0.1,0.5,0.75,0.8,0.85,0.9,0.95,1]
  # parameter         = 'mutation_prob'
  # parameter_values  = ['1/n','5/n','10/n','15/n','20/n','25/n']
  parallelism       = max(1, os.cpu_count() - 3) # leave 3 cores free for OS

  print(f"""
  wdimacs_instances     {len(wdimacs_instances)} ({wdimacs_instances})
  parameter             '{parameter}'
  parameter_values      {len(parameter_values)} ({parameter_values})

  time_budget           {time_budget}s
  repetitions           {repetitions}

  parallelism           {parallelism}

  total execution time  {(((len(wdimacs_instances)*len(parameter_values)) * (time_budget*repetitions)) / parallelism):.0f}s
  """)

  # Create a figure with subplots for each instance
  fig, axes = plt.subplots(1, len(wdimacs_instances), figsize=(5*len(wdimacs_instances), 5), constrained_layout=True)
  fig.set_size_inches(5 * len(wdimacs_instances), 5)

  info_text = ', '.join([ f"{k}: {v}" for k,v in vars(args).items() if v is not None ])
  fig.suptitle(info_text, fontsize=9, family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

  wdimacs_parsed = {}
  wdimacs_runs = {}

  print(f"instance\t'{parameter}'\trepetitions\ttime\tt_avg\tnsat_avg")

  # Pre-initialisation loop
  for idx, wdimacs in enumerate(wdimacs_instances):

    # Check file exists
    if not os.path.isfile(f"{instance_folder}/{wdimacs}"):
      raise Exception(f"file '{instance_folder}/{wdimacs}' doesn't exist")
    
    wdimacs_parsed[wdimacs] = parse_wdimacs(f"{instance_folder}/{wdimacs}")
    n,m,clauses = wdimacs_parsed[wdimacs]
    
    # Initialize empty runs with all parameter values
    wdimacs_runs[wdimacs]   = {str(pv): [] for pv in parameter_values}

    # Pre-populate graphs with structure
    axes[idx].boxplot(wdimacs_runs[wdimacs].values(), tick_labels=wdimacs_runs[wdimacs].keys())
    axes[idx].set_title(f"{wdimacs}\n(vars={n},clauses={m})", fontsize=9)
    axes[idx].set_xlabel(parameter)
    axes[idx].set_ylabel("nsat")

  # Main loop
  for idx, wdimacs in enumerate(wdimacs_instances):

    n,m,clauses = wdimacs_parsed[wdimacs]

    real_parameter_values = [eval(v) if isinstance(v, str) else v for v in parameter_values]

    for j,parameter_value in enumerate(parameter_values):

      real_parameter_value = real_parameter_values[j]

      # Run repetitions of evolutionary_algorithm in parallel
      func_args   = [n, m, clauses, int(time_budget)]
      func_kwargs = {**kwargs,parameter:real_parameter_value}
      start = time.perf_counter()
      result = run_parallel(
        evolutionary_algorithm,
        int(repetitions),
        func_args, func_kwargs,
        parallelism
      )
      end = time.perf_counter()

      nsat_log = [int(nsat) for _, nsat, _, *_ in result]
      t_log    = [int(t)    for t, _, _, *_ in result]
      wdimacs_runs[wdimacs][f"{parameter_value}"] = nsat_log

      print(f"{wdimacs}\t{parameter_value}\t{repetitions}\t{(end-start):.6f}s\t{np.array(t_log).mean()}\t{np.array(nsat_log).mean()}")

      # Update boxplot interactively
      axes[idx].clear()
      axes[idx].boxplot(wdimacs_runs[wdimacs].values(), tick_labels=wdimacs_runs[wdimacs].keys())
      axes[idx].set_title(f"{wdimacs}\n(vars={n},clauses={m})", fontsize=9)
      axes[idx].set_xlabel(parameter)
      axes[idx].set_ylabel("nsat")
      # axes[idx].axhline(y=m, linestyle="--", linewidth=1, color="#008000")
      plt.pause(0.1)

    # draw boxplot of results
    if verbose >= 1:
      print(wdimacs_runs[wdimacs])

  print(wdimacs_runs)
  print("done!")

  # Construct file name
  wdimacs_instances_safe = [re.sub('^(?:.*/)?(.*)\\..*','\\1',instance) for instance in wdimacs_instances]
  base = f"{parameter}_{time_budget}s_{repetitions}x_{'_'.join(wdimacs_instances_safe)}"
  filename = f"{base}.png"
  counter = 1
  while os.path.exists(filename):
    filename = f"{base}_{counter}.png"
    counter += 1

  output_folder_path = f"{output_folder}/{parameter}"
  output_file_path   = f"{output_folder_path}/{filename}"
  os.makedirs(output_folder_path, exist_ok=True)
  fig.set_size_inches(5 * len(wdimacs_instances), 5)
  fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
  print(f"saved to {output_file_path}")

  plt.ioff()
  plt.show()
