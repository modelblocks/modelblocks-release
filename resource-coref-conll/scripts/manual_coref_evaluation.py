'''
Compares two CONLL-formatted coreference files for 1. anaphoricity recall/precision and 2. among correctly recalled mentions, what is precision of correct antecedent choice? 

Usage:
    python <script-name> key.conll preds.conll parsed.tokdecs
'''


import sys
import pdb
DELIM = "\t"
TOKDECDELIM = " "
PRONOUNS = ["He","he","She","she","They","they","It","it","Him","him","Her","her","Them","them","His","his","Their","their","Its","its","I","i","Me","me","My","my","Himself","himself","Herself","herself","Myself","myself","Themselves","themselves","We","we","Us","us","Our","our"]

#I/O
with open(sys.argv[1], 'r') as gold_fh: #get gold conll
    goldlines = gold_fh.readlines()
goldlines = [x for x in goldlines if not x.startswith("#end")]

with open(sys.argv[2], 'r') as predicted_fh: #get predicted conll
    predictedlines = predicted_fh.readlines()
predictedlines = [x for x in predictedlines if not x.startswith("#end")]
assert len(goldlines) == len(predictedlines)

with open(sys.argv[3],'r') as tokdecsfh: #get tokdecs
    tokdecs = tokdecsfh.readlines()
tokdecs = tokdecs[1:] #remove header, resulting in only article delim
assert len(tokdecs) == len(goldlines) #should be equal if header removed (and enddoc lines removed in conllfiles), leaving only conll begin doc lines to match tokdecs !ARTICLE delims

#anaphoricity recall, precision. 
#recall - for all gold mentions, how many are predicted as mentions?
#precision - for predicted mentions, how many are mentions in gold?  
predcount = 0
goldcount = 0
predcorrect = 0

goldprocount = 0
propredcount = 0
propredcorrect = 0

golduppercount = 0
upperpredcount = 0
upperpredcorrect = 0

goldlowercount = 0
lowerpredcount = 0
lowerpredcorrect = 0

predchains = {} #{doc:{cid:[uid,...],cid2:[uid2,...],...}, doc2:{cid2:[uid3,...]}, ...}
goldchains = {} #dict of dict of lists
docnum = 0 

for i,goldline in enumerate(goldlines):
    if goldline.startswith("#"):
        docnum += 1
        continue
    predline = predictedlines[i]
    try:
        _,_,_,_,_,goldcid = goldline.split(DELIM)
        word,_,_,_,predcid = predline.split(DELIM)
    except:
        print(goldline)
        print(predline)
    if predcid != "-\n":
        predcount += 1
        if docnum not in predchains:
            predchains[docnum] = {}
        if predcid not in predchains[docnum]:
            predchains[docnum][predcid] = []
        predchains[docnum][predcid].append(i)
        if goldcid != "-\n":
            predcorrect += 1
    if goldcid != "-\n":
        goldcount += 1
        if docnum not in goldchains:
            goldchains[docnum] = {}
        if goldcid not in goldchains[docnum]:
            goldchains[docnum][goldcid] = []
        goldchains[docnum][goldcid].append(i) #confirmed cids not resued across different documents in gold

    if word in PRONOUNS:
        goldprocount += 1
        if predcid != "-\n":
            propredcount += 1
            if goldcid != "-\n":
                propredcorrect += 1

    if word.isupper():
        golduppercount += 1
        if predcid != "-\n":
            upperpredcount += 1
            if goldcid != "-\n":
                upperpredcorrect += 1

    if word.islower():
        goldlowercount += 1
        if predcid != "-\n":
            lowerpredcount += 1
            if goldcid != "-\n":
                lowerpredcorrect += 1
            
def f1(prec,rec):
    return (2*prec*rec)/(prec+rec)

anaphoricity_recall = predcorrect / float(goldcount)
anaphoricity_precision = predcorrect / float(predcount)
anaph_f1 = f1(anaphoricity_recall,anaphoricity_precision)
pro_rec = propredcorrect / float(goldprocount)
pro_prec = propredcorrect / float(propredcount)
pro_f1 = f1(pro_rec,pro_prec)
upp_rec = upperpredcorrect / float(golduppercount)
upp_prec = upperpredcorrect / float(upperpredcount)
upp_f1 = f1(upp_rec,upp_prec)
low_rec = lowerpredcorrect / float(goldlowercount)
low_prec = lowerpredcorrect / float(lowerpredcount)
low_f1 = f1(low_rec,low_prec)

print("Pronoun prec: {} recall: {}, f1: {}".format(pro_prec,pro_rec,pro_f1))
print("Anaphoricity prec: {}, rec: {}, f1: {}".format(anaphoricity_precision, anaphoricity_recall, anaph_f1))      
print("Upper prec: {} recall: {}, f1: {}".format(upp_prec, upp_rec, upp_f1))
print("Lower prec: {} recall: {}, f1: {}".format(low_prec,low_rec, low_f1))

#Calculate antecedent precision of correctly recalled mentions - of the correctly recalled mentions, what percentage have the correct antecedent chosen?

#build gold negative offset chains by mentionid (linenum)
#some of these are empty - e.g. {2:[],...}  - because it's the list of all valid prior antecedents and the first mention doesn't have any priors.
mid2offsets = {} #get offset from goldchains, keyed by mention id. e.g., {234:[-2,-15,...],5000:[-7],...}
for docid in goldchains:
    for goldcid in goldchains[docid]:
        for mid in goldchains[docid][goldcid]:
            mid2offsets[mid] = [-(mid-x) for x in goldchains[docid][goldcid] if mid > x] 

#score antecedent choices with strict (most recent) and relaxed (any correct antecedent in chain - matches training)
antecorrect = 0 #relaxed
santecorrect = 0 #strict 

proantecorrect = 0 #relaxed, pronouns with correctly id'd antecedent idx (any of them in chain)
propredcorrect = 0 #pronoun correct mention detection - denom.  all correctly mentionid'd pronouns.
for i,tdline in enumerate(tokdecs):
    if tdline.startswith("!ARTICLE"):
        continue
    _,_,_,_,_,goldcid = goldlines[i].split(DELIM)
    _,_,_,_,predcid = predictedlines[i].split(DELIM)
    if predcid != "-\n" and goldcid != "-\n": #limit to correct mention detection cases 
        word,_,fdec,jdec,_,predoffset,surp = tdline.split(TOKDECDELIM) #get offset from tokdecs as predicted offset
        predoffset = int(predoffset) 
        if word in PRONOUNS:
            propredcorrect += 1
            if predoffset in mid2offsets[i]:
                proantecorrect += 1
        #subset to tokdecs making a non-zero prediction - don't count first mentions as having a valid antecedent. or should this be changed to also count first mentions that correctly chose null antecedent?
        if predoffset != 0:
            if predoffset in mid2offsets[i]: #relaxed correctness is any previous in chain (matches training)
                antecorrect += 1
            try:
                if mid2offsets[i] != []: #only check for correctness if gold has any correct antecedents.  if predoffset is not zero but gold has zero correct antecedents, prediction is wrong - no need to check min.
                    if predoffset == min(mid2offsets[i]): #strict correctness is most recent (smallest negative offset)
                        santecorrect += 1
            except:
                pdb.set_trace()

relaxed_acc = float(antecorrect) / predcorrect
strict_acc = float(santecorrect) / predcorrect
pro_acc = float(proantecorrect) / propredcorrect

print("For correct mention detections, accuracy of antecedent choice where relaxed is any correct antecedent in chain, while strict is most recent antecedent mention in chain")
print("Relaxed pronoun acc: {} relaxed acc: {}".format(pro_acc,relaxed_acc)) #Strict_acc: {}".format(relaxed_acc, strict_acc))
