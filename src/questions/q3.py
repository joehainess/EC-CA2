from src.common.maxsat import parse_wdimacs,evolutionary_algorithm
import cProfile
import matplotlib.pyplot as plt

def run(wdimacs, time_budget, repetitions, **kwargs):

  n,m,clauses = parse_wdimacs(wdimacs)

  verbose = kwargs.get('verbose') or 0
  graph   = kwargs.get('graph')   or False
  boxplot = kwargs.get('boxplot') or False

  ax = None
  if graph:
    # start graph
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best N Sat")
    ax.axhline(y=m, linestyle="--", linewidth=1, color="#008000")
    ax.set_title("Evolutionary Algorithm Progress")

  nsat_log = []
  for i in range(0,int(repetitions)):

    # create line for EA to draw on
    line = None
    if ax:
      alpha = max(0.2, (1/int(repetitions)))
      line, = ax.plot([], [], color="red", linestyle="-", linewidth=1, alpha=alpha)
    
    # run EA
    t,nsat,xbest,*_ = evolutionary_algorithm(n, m, clauses, int(time_budget), **{**kwargs,'ax':ax,'line':line})
    
    nsat_log.append(int(nsat))
    print(
      '\t'.join([str(t),str(nsat),''.join(map(str, xbest))])
    )
  
  # remove graph
  if ax:
    plt.ioff()
    # plt.close('all')
    plt.show()
  
  # draw boxplot of results
  if boxplot:
    if verbose >= 1:
      print(nsat_log)
    plt.boxplot([nsat_log], labels=["1"], vert=False)
    plt.draw()
    plt.show()
  
  return None