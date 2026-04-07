from src.common.maxsat import parse_clause_str,parse_assignment_str,sat_check


# Exercise 1. (10 % of the marks) Implement a satisfiability checker which given a clause and an
# assignment determines whether the clause is satisfied by the assignment.
# A clause is represented by a sequence of values, where you can ignore the first value. Each of the
# next values is either a positive integer, to indicate a positive literal, or a negative integer, indicating
# a negative literal. The indices of the literal start from 1, and not 0. The representation of a clause
# is completed by a integer 0. As an example, the string 0.5 2 1 -3 -4 0 represents the clause
# (x2 ↑ x1 ↑ ¬x3 ↑ ¬x4).

def run(clause_str, assignment_str):

  clause     = parse_clause_str(clause_str)
  assignment = parse_assignment_str(assignment_str)

  return sat_check(clause, assignment)