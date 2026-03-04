import argparse
import sys

from common.maxsat import parse_wdimacs,evolutionary_algorithm
import matplotlib.pyplot as plt

# Enable interactive mode for live updating plots
plt.ion()

prog_name = sys.argv[0]

with open("/tmp/invokations.log", "a") as myfile:
    myfile.write(' '.join(map(str, sys.argv)) + "\n")

parser = argparse.ArgumentParser(prog=prog_name)
# parser.add_argument('-question',    required=True, choices=['3'])
# parser.add_argument('-wdimacs',     required=True)
parser.add_argument('-time_budget', required=True)
parser.add_argument('-repetitions', required=True)

# parser.add_argument('-population_size')
# parser.add_argument('-crossover_prob')
# parser.add_argument('-mutation_prob')

parser.add_argument('-verbose','-v', action='store_true')

args = parser.parse_args()

time_budget, repetitions = (vars(args)[k] for k in ['time_budget', 'repetitions'])
kwargs = {k: v for k, v in vars(args).items() if k not in ('time_budget', 'repetitions')}

time_budget = int(time_budget)
repetitions = int(repetitions)



wdimacs_instances = ['c1335.wcnf','c3540.wcnf']
parameter         = 'population_size'
parameter_values  = [25,50,100,150,200,250]
# parameter         = 'mutation_prob'
# parameter_values  = ['1/n','2/n','3/n','4/n','5/n']

print(f"""
wdimacs_instances     {len(wdimacs_instances)} ({wdimacs_instances})
parameter             '{parameter}'
parameter_values      {len(parameter_values)} ({parameter_values})
time_budget           {time_budget}s
repetitions           {repetitions}

total execution time  {(len(wdimacs_instances)*len(parameter_values)) * (time_budget*repetitions)}s
""")

print(f"instance\t'{parameter}'\trepetition\tt\tnsat")

# Create a figure with subplots for each instance
fig, axes = plt.subplots(1, len(wdimacs_instances), figsize=(10, 5))
# fig.suptitle(f" time_budget={time_budget}s\nrepetitions={repetitions}")
plt.subplots_adjust(top=0.75)  # make room at top

info_text = (f"""
time_budget: {time_budget}s
repetitions: {repetitions}
""")
fig.text(0.5, 0.95, info_text,
         ha='center', va='top', fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

wdimacs_runs = {}

# Pre-populate graphs with structure
for idx, wdimacs in enumerate(wdimacs_instances):
  # Initialize empty runs with all parameter values
  wdimacs_runs[wdimacs] = {str(pv): [] for pv in parameter_values}
  axes[idx].boxplot(wdimacs_runs[wdimacs].values(), tick_labels=wdimacs_runs[wdimacs].keys())
  axes[idx].set_title(f"{wdimacs}")
  axes[idx].set_xlabel(parameter)
  axes[idx].set_ylabel("nsat")

for idx, wdimacs in enumerate(wdimacs_instances):

  n,m,clauses = parse_wdimacs(wdimacs)

  real_parameter_values = [eval(v) if isinstance(v, str) else v for v in parameter_values]

  for j,parameter_value in enumerate(parameter_values):

    real_parameter_value = real_parameter_values[j]

    verbose = vars(args).get('verbose') or False

    nsat_log = []
    for i in range(0,int(repetitions)):

      # run EA
      t,nsat,xbest = evolutionary_algorithm(n, m, clauses, int(time_budget),
                                            **{**kwargs,parameter:real_parameter_value})
      nsat_log.append(int(nsat))
      wdimacs_runs[wdimacs][f"{parameter_value}"] = nsat_log

      print(f"{wdimacs}\t{parameter_value}\t{i+1}/{repetitions}\t{t}\t{nsat}/{m}")

      # Update boxplot interactively
      axes[idx].clear()
      axes[idx].boxplot(wdimacs_runs[wdimacs].values(), tick_labels=wdimacs_runs[wdimacs].keys())
      axes[idx].set_title(f"{wdimacs}")
      axes[idx].set_xlabel(parameter)
      axes[idx].set_ylabel("nsat")
      plt.pause(0.1)

  # draw boxplot of results
  if verbose:
    print(wdimacs_runs[wdimacs])

print("done!")
plt.ioff()
plt.show()
