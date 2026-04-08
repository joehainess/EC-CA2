from src.common.maxsat import parse_wdimacs,parse_assignment_str,n_sat

def run(wdimacs, assignment_str):

  n,m,clauses = parse_wdimacs(wdimacs)
  assignment  = parse_assignment_str(assignment_str)
  
  return n_sat(clauses, assignment)
