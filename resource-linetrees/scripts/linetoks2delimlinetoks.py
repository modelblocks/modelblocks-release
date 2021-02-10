import sys

'''
Prepend article delim !ARTICLE to linetoks to create delim.linetoks.  Do nothing if !ARTICLE begins linetoks
'''

firstline = sys.stdin.readline()
if firstline != "!ARTICLE\n":
    sys.stdout.write("!ARTICLE\n")
sys.stdout.write(firstline)
for line in sys.stdin.readlines():
    sys.stdout.write(line)
