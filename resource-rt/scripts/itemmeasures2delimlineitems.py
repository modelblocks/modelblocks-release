import sys
'''
Convert itemmeasures to delim.lineitems, where each line is an space-delimited sentence, one per line, and article delimiter (!ARTICLE) are inserted between articles.  Don't insert initial article delim, as it is inserted when converting later to linetoks.
'''
ARTICLEDELIM = "!ARTICLE"
docid = ""
sentid = ""
senttoks = []

#assert sys.stdin.readline() == "word sentid sentpos docid\n"
sys.stdin.readline() #strip header, allow for files without docid

#handle remaining lines, outputting lineitems and inserting !ARTICLE when docid changes
for line in sys.stdin.readlines():
    #item, newsentid, sentpos, newdocid = line.strip().split(" ")
    fields = line.strip().split(" ")
    item, newsentid = fields[:2]
    newdocid = fields[3] if (len(fields) == 4) else "default" #handling for itemmeasures without docid column
    if sentid != newsentid and senttoks != []:
        sys.stdout.write(" ".join(senttoks)+"\n") #output sentence
        senttoks = [] #reset senttoks
    sentid = newsentid #update sentid
    if docid != newdocid:
        sys.stdout.write(ARTICLEDELIM+"\n")
        docid = newdocid
    senttoks += [item]

#final sentence output
if senttoks != []:
    sys.stdout.write(" ".join(senttoks)+"\n")


    





