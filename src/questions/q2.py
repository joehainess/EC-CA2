from src.common.maxsat import parse_wdimacs,parse_assignment_str,n_sat

# Exercise 2. (10 % of the marks) Implement a routine for importing MAXSAT instances on the
# WDIMACS file format. A WDIMACS file is a text file where each line either
# • begins with the letter c, indiciating that the rest of the line is a comment
# • begins with p cnf n m x where n is the number of variables, m is the number of clauses, and
# x is an optional parameter which may or may not be present, or
# • describes a clause, as explained in Exercise 1.
# To test that the import routine works correctly, you should count the number of clauses which are
# satisfied by a given assignment.

def run(wdimacs, assignment_str):

  n,m,clauses = parse_wdimacs(wdimacs)
  assignment  = parse_assignment_str(assignment_str)
  
  return n_sat(clauses, assignment)
