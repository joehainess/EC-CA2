import argparse
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import time
import numpy as np
import re
from src.common.maxsat import parse_wdimacs,evolutionary_algorithm
import matplotlib.pyplot as plt


def run_parallel_with_queue(fn, n, args, kwargs, max_workers, cb):
    """
    Run fn(*args, **kwargs) n times in parallel.
    Returns a list of n results in completion order.
    Invokes the cb function every time the generation record is updated (via Queue puts).
    """
    results = [None] * n

    gen_record = {}

    with Manager() as manager:
      queue = manager.Queue()

      with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, *args, **{**kwargs,'queue':queue}): i for i in range(n)}

        done_count = 0
        while done_count < len(futures):
          while not queue.empty():
            gen,nsat = queue.get()
            if gen not in gen_record:
              gen_record[gen] = []
            gen_record[gen].append(nsat)
            cb(gen_record)
          
          for future in futures:
            if future.done() and not hasattr(future, "_counted"):
              future._counted = True
              i = futures[future]
              results[i] = future.result()
              done_count = done_count + 1
        
        time.sleep(0.1)

    return results


if __name__ == '__main__':

  # Enable interactive mode for live updating plots
  plt.ion()

  prog_name = sys.argv[0]

  parser = argparse.ArgumentParser(prog=prog_name)
  parser.add_argument('-wdimacs',     required=True)
  parser.add_argument('-time_budget', required=True)
  parser.add_argument('-repetitions', required=True)

  parser.add_argument('-max_generations')
  parser.add_argument('-verbose','-v', action='count')

  args = parser.parse_args()

  wdimacs, time_budget, repetitions = (vars(args)[k] for k in ['wdimacs', 'time_budget', 'repetitions'])
  kwargs = {k: v for k, v in vars(args).items() if k not in ('wdimacs', 'time_budget', 'repetitions')}

  verbose = kwargs.get('verbose') or 0
        
  time_budget = int(time_budget)
  repetitions = int(repetitions)

  output_folder     = 'plots/parameter_sweep'

  # parameter         = 'lambda_mu_ratio'
  # parameter_values  = [2,3,4,5,6,7,8,9,10]
  # parameter         = 'crossover_prob'
  # parameter_values  = [0,0.1,0.5,0.75,0.8,0.85,0.9,0.95,1]
  parameter         = 'mutation_prob'
  parameter_values  = ['1/n','5/n','10/n','15/n','20/n','25/n']

  parallelism       = max(1, os.cpu_count() - 3) # leave 3 cores free for OS
  colors            = ["#DBA714","#391FEA","#E5154B","#3E9A71","#3495F5","#DB5E3A","#46385C","#8625B8","#83CA5D","#6252A1","#57510C","#D9B182",]

  print(f"""
  wdimacs               {wdimacs}
  parameter             '{parameter}'
  parameter_values      {len(parameter_values)} ({parameter_values})

  time_budget           {time_budget}s
  repetitions           {repetitions}
  parallelism           {parallelism}

  total execution time  {((len(parameter_values) * (time_budget*repetitions)) / parallelism):.0f}s
  """)

  n,m,clauses = parse_wdimacs(f"{wdimacs}")
  real_parameter_values = [eval(v) if isinstance(v, str) else v for v in parameter_values]

  # start graph
  plt.ion()
  fig, ax = plt.subplots(1, 1, figsize=(5+1, 5), constrained_layout=True)
  info_text = f"time_budget: {time_budget}, repetitions: {repetitions}"
  fig.suptitle(info_text, fontsize=9, family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

  ax.set_xlabel("Generation")
  ax.set_ylabel("Best N Sat")
  # ax.axhline(y=m, linestyle="--", linewidth=1, color="#008000")
  ax.set_title(f"{wdimacs}\n(vars={n},clauses={m})", fontsize=9)

  for j,parameter_value in enumerate(parameter_values):

    real_parameter_value = real_parameter_values[j]
    color = 'black' if j >= len(colors) else colors[j]

    fill = None
    avg_line = None
    last_updated = time.perf_counter()

    def cb(gen_record):
      """
        Updates plot at regular intervals 
      """
      global fill, avg_line, last_updated
      if time.perf_counter() < last_updated + 0.25:
        return
      last_updated = time.perf_counter()
      gens = []
      gen_avgs = []
      gen_mins = []
      gen_maxs = []
      for gen,nsats in gen_record.items():
        gens.append(gen)
        gen_avgs.append(sum(nsats) / len(nsats))
        gen_mins.append(min(nsats))
        gen_maxs.append(max(nsats))

      # Plot avg line
      if avg_line is None:
        avg_line, = ax.plot([],[], color=color, linestyle="-", linewidth=1.5, alpha=0.8, label=f"{parameter}={parameter_value}")
        ax.legend()
      avg_line.set_xdata(gens)
      avg_line.set_ydata(gen_avgs)
  
      # Plot min/max fill
      if fill is not None:
        fill.remove()
      fill = ax.fill_between(gens, gen_mins, gen_maxs, color=color, alpha=0.1)
      
      plt.pause(0.01)

    real_parameter_value = real_parameter_values[j]

    # Run repetitions of evolutionary_algorithm in parallel
    func_args   = [n, m, clauses, int(time_budget)]
    func_kwargs = {**kwargs,parameter:real_parameter_value}
    result = run_parallel_with_queue(
      evolutionary_algorithm,
      int(repetitions),
      func_args, func_kwargs,
      os.process_cpu_count() - 3,
      cb
    )

  print("done!")

  # Construct file name
  base = f"{parameter}_{time_budget}s_{repetitions}x_{re.sub('^(?:.*/)?(.*)\\..*','\\1',wdimacs)}"
  filename = f"{base}.png"
  counter = 1
  while os.path.exists(filename):
    filename = f"{base}_{counter}.png"
    counter += 1

  output_folder_path = f"{output_folder}/{parameter}"
  output_file_path   = f"{output_folder_path}/{filename}"
  os.makedirs(output_folder_path, exist_ok=True)
  fig.set_size_inches(5+1, 5)
  fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
  print(f"saved to {output_file_path}")

  plt.ioff()
  plt.show()
