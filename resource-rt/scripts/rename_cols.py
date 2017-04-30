import sys, argparse

argparser = argparse.ArgumentParser('''
Reads a space-delimited data table and renames user-specified columns. Column source and target names must alternate: col_1_source col_1_target [col_2_source col_2_target [etc...
''')
argparser.add_argument('cols', nargs='+', type=str, help='column names to output')
args=argparser.parse_args()
assert len(args.cols) % 2 == 0, 'Odd-numbered sequence of column names disallowed (source names and target names must alternate)'

def main():
    src =  args.cols[::2]
    targ = args.cols[1::2]
    header = True
    line = sys.stdin.readline()
    while line:
        if header:
            HEAD = line.strip().split()
            for i,c in enumerate(src):
                ix = HEAD.index(c) if c in HEAD else None
                if ix != None:
                    HEAD[ix] = targ[i]
            sys.stdout.write(' '.join(HEAD) + '\n')
            header = False
        else:
            sys.stdout.write(line)
        line = sys.stdin.readline()

main()
