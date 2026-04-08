from src.common.maxsat import parse_clause_str,parse_assignment_str,sat_check

def run(clause_str, assignment_str):

  clause     = parse_clause_str(clause_str)
  assignment = parse_assignment_str(assignment_str)

  return sat_check(clause, assignment)