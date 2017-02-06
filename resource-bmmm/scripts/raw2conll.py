import sys

for line in sys.stdin:
    ix = 1
    for word in line.split():
        print(str(ix) + '\t' + word + '\t_\t_\t_\t_\t_\t_\t_')
	ix += 1
    print('')
