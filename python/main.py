import argparse
import sys

import questions.q1
import questions.q2
import questions.q3

prog_name = sys.argv[0]

with open("/tmp/invokations.log", "a") as myfile:
    myfile.write(' '.join(map(str, sys.argv)) + "\n")

parser = argparse.ArgumentParser(prog=prog_name)
parser.add_argument('-question', required=True, choices=['1', '2', '3'])
args, unknown = parser.parse_known_args()

# Switch case for question handling
match args.question:
    case '1':
        q1parser = argparse.ArgumentParser(prog=prog_name)
        q1parser.add_argument('-question',   required=True, choices=['1'])
        q1parser.add_argument('-clause',     required=True)
        q1parser.add_argument('-assignment', required=True)
        q1args = q1parser.parse_args()

        result = questions.q1.run(q1args.clause, q1args.assignment)
        if result != None:
          print(result)

    case '2':
        q2parser = argparse.ArgumentParser(prog=prog_name)
        q2parser.add_argument('-question',   required=True, choices=['2'])
        q2parser.add_argument('-wdimacs',    required=True)
        q2parser.add_argument('-assignment', required=True)
        q2args = q2parser.parse_args()

        result = questions.q2.run(q2args.wdimacs, q2args.assignment)
        if result != None:
          print(result)

    case '3':
        q3parser = argparse.ArgumentParser(prog=prog_name)
        q3parser.add_argument('-question',    required=True, choices=['3'])
        q3parser.add_argument('-wdimacs',     required=True)
        q3parser.add_argument('-time_budget', required=True)
        q3parser.add_argument('-repetitions', required=True)

        q3parser.add_argument('-population_size')
        q3parser.add_argument('-crossover_prob')
        q3parser.add_argument('-mutation_prob')
        q3parser.add_argument('-parents_ratio')
        q3parser.add_argument('-offspring_ratio')

        q3parser.add_argument('-verbose','-v', action='store_true')
        q3parser.add_argument('-graph','-g',   action='store_true')
        q3parser.add_argument('-boxplot','-b', action='store_true')
        q3args = q3parser.parse_args()

        wdimacs, time_budget, repetitions = (vars(q3args)[k] for k in ['wdimacs', 'time_budget', 'repetitions'])
        kwargs = {k: v for k, v in vars(q3args).items() if k not in ('wdimacs', 'time_budget', 'repetitions')}

        result = questions.q3.run(wdimacs, time_budget, repetitions, **kwargs)
        if result != None:
          print(result)

    case None | _:
        parser.print_help()
