'''
Title: extract_coref_predictors.py
Author: Evan Jaffe
Date: 3/28/2017

Generate by-word tsv of predictors for coreference-based features. Uses annotated numbered editable linetrees file, with X-nID and X-mID as indicating the id of the antecedent.

To run:
    python extract_coref_predictors.py <input_filename>

'uid', 'corefbin', 'coreflenw', 'coreflenr', 'corefsize'

Output format:
Column 1: word_id, in sentenceid_wordid format
Column 2: binary corefence indicator
Column 3: distance in words to antecedent
Column 4: distance in nouns,verbs to antecedent
Column 5: size of chain in mention count up to current mention
'''
from __future__ import division, print_function
import sys
import re
import pdb

class CorefPredictors:

    def __init__(self, lines):
        self.lines = lines
        self.chainsize = {} #{last_mention_id:mentioncount} 
        self.ref_cats = ['N','V','G','B','L'] #noun, finite-verb, gerund, base-form, participle
        self.predictors = []

    def get_cat(self, line):
        #find first letter of nearest cat to word
        #pdb.set_trace()
        match = re.search(r"\((\S*)\s\S+\)",line) #cap,non-space,space,word
        try:
            cat = match.group(1)[0] #first character of cat
        except AttributeError:
            pdb.set_trace()
            cat = 'punc'
            print(line)
        return cat

    def get_dist(self, mtype, curr_id, ante_id):
        '''type is 'word' or 'ref' '''
        dist = 0
        on = False
        for line in self.lines:
                line_id = line.split(":")[0]
                cat = self.get_cat(line)
                if ante_id == line_id:
                    on = True
                    continue 
                if on == True:
                    if mtype == 'word':
                        dist += 1
                    elif mtype == 'ref':
                        if cat in self.ref_cats:
                            dist += 1
                if curr_id == line_id:
                    #assert dist > 0 #else cataphor. could run over reversed lines to get cataphors, but i think only one or two are in the corpus?
                    #TODO fix for cataphors
                    return dist
                
    def get_chain_size(self, curr_id, ante_id):
        if ante_id not in self.chainsize:
            self.chainsize[curr_id] = 1
        else:
            self.chainsize[curr_id] = self.chainsize[ante_id] + 1
        return self.chainsize[curr_id]

    def get_predictors(self):
        for idx, line in enumerate(self.lines):
            #if idx % 100 == 0:
                #print("Processing line {0}".format(idx))
            #if idx % 100 == 0 and idx > 1: #testing only - REMOVE THESE TWO LINES
            #    break
            curr_id = line.split(":")[0] #current word id group
            match = re.search(r".*-[nm]([0-9]+) ", line)
            if match is not None:
                ante_id = match.group(1) # antecedent id group
                binary_coref_indic = 1
                word_dist = self.get_dist('word', curr_id, ante_id)
                ref_dist = self.get_dist('ref', curr_id, ante_id)
                chain_size = self.get_chain_size(curr_id, ante_id)
                self.predictors.append([curr_id, str(binary_coref_indic), str(word_dist), str(ref_dist), str(chain_size)]) #list of lists
            else:
                binary_coref_indic = 0
                self.predictors.append([curr_id, str(binary_coref_indic), "0", "0", "0",])

    def print_predictors(self):
        #pdb.set_trace()
        print("\t".join(['uid', 'corefbin', 'coreflenw', 'coreflenr', 'corefsize']))
        for p in self.predictors:
            print("\t".join(p))

def main():
    #print("Beginning feature extraction...")
    with open(sys.argv[1], 'r') as iff:
        lines = iff.readlines()

    cp = CorefPredictors(lines)
    cp.get_predictors()
    cp.print_predictors()

if __name__ == "__main__":
    main()
