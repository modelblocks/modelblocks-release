#! /usr/bin/python
import sys
import re

'''
Print warnings if predictors are not found in datatable
'''
def removez(mstr):
    match = re.match("z\.\((.*)\)", mstr)
    if match is not None:
        return match.group(1)
    else:
        return mstr

def formula2predictorstr(lines):
    #e.g. fdur ~ z.(myfixed) + (1 | z.(myrand))
    preds = []
    for line in lines:
        fields = line.split(" ")
        #remove ~,|,(1 etc.
        fields = [x for x in fields if (x not in ["~","(","(1","|","subject)",")","+"])]
        #remove z.() circumfix
        fields = [removez(x) for x in fields]
        #strip leading/trailing space, paren
        fields = [x.strip() for x in fields]
        #return list of preds as strings
        preds += fields
    return preds

# tblfile, bform, preds_add, preds_ablate = sys.argv[1:]
# 
# #get all predictors - strip z.() circumfixes
# with open(bform,'r') as iff:
#     lines = iff.readlines()
# allpreds = formula2predictorstr(lines)
# allpreds += [preds_ablate]
# 
# #open tblfile and get header for colnames
# with open(tblfile, 'r') as iff:
#     header = iff.readline()
#     cols = header.strip().split(" ")
# 
# #compare all predictors to colnames - warn if any missing
# for pred in allpreds:
#     if pred not in cols:
#         print("Warning! Could not find predictor {} in dataframe".format(pred))
#         sys.stderr.write("Warning! Could not find predictor {} in dataframe\n".format(pred))
