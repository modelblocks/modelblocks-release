'''
Title: extract_coref_predictors.py
Author: Evan Jaffe
Date: 3/28/2017

Generate by-word tsv of predictors for coreference-based features. Uses annotated numbered editable linetrees file, with X-nID and X-mID as indicating the id of the antecedent.

To run:
    python extract_coref_predictors.py <input_filename>

"uid", "corefbin", "coreflenw", "coreflenr", "corefsize", "isanaphpro", "coreflenwlog", "coreflenrlog", "corefsizelog"

Output format:
Column 1: word_id, in sentenceid_wordid format
Column 2: binary corefence indicator
Column 3: distance in words to antecedent
Column 4: distance in nouns,verbs to antecedent
Column 5: size of chain in mention count up to current mention
'''
from __future__ import division, print_function
from math import log
import sys
import re
import pdb

DELIM = " "

def log1p(val):
    return log(1+val)

class CorefPredictors:

    def __init__(self, lines):
        self.lines = lines
        self.chainsize = {} #{last_mention_id:mentioncount} 
        self.ref_cats = ['N','V','G','B','L'] #noun, finite-verb, gerund, base-form, participle
        self.pronouns = ['It', 'it', 'he', 'He', 'she', 'She', 'They', 'they', 'them', 'Them', 'I', 'her', 'Her', 'Him', 'him', 'Himself', 'himself', 'herself', 'Herself', 'themselves', 'Themselves', 'We', 'we', 'this','This', 'that','That','those','Those', 'You','you','us','Us', 'itself','Itself', 'myself', 'Myself', 'Ourselves', 'ourselves', 'That', 'that'] #doesn't include some that we didn't mark, relative pronouns, interrogative pronouns, etc.  also, some of these included have ambiguous POS (e.g., nominal, relativizer or determiner 'that', nominal or determiner 'that'). but that's not a problem since we're only subsetting pronouns out of mentions already marked as anaphoric.
        self.predictors = []
        self.storyidxes = {"bradfordboar":(1,57), "aqua":(58,95), "matchstickgirl":(96,149), "birdking":(150,204), "elvis":(205,249), "mrsticky":(250,313), "textmsg":(314,361), "ufo":(362,394), "tulipmania":(395,442), "tourettes":(443,485)} #{storyid:(startidx,endidx), ...}

    def sentidx2storyid(self, sentidx):
        #self.sentidx2storyid = {} #{idx:storyid, ...}
        for story in self.storyidxes:
            if sentidx in range(self.storyidxes[story][0], self.storyidxes[story][1]+1):
                return story
        return None

    def is_pro(self, word):
        return True if word in self.pronouns else False

    def get_word(self, line):
        #pdb.set_trace()
        #get word from line
        match = re.search(r"\s([^\) ]+)\)",line) #space, anything not a space or close paren, close paren
        try:
            word = match.group(1)
        except AttributeError:
            pdb.set_trace()
        return word

    def get_full_cat(self, line):
        #find full category of word
        match = re.search(r"\((\S*)\s\S+\)",line) #cap,non-space,space,word
        return match.group(1)

    def get_cat(self, line):
        #find first letter of nearest cat to word
        #pdb.set_trace()
        match = re.search(r"\((\S*)\s\S+\)",line) #cap,non-space,space,word
        try:
            cat = match.group(1)[0] #first character of cat
        except AttributeError:
            pdb.set_trace()
            cat = "punc"
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
                    if mtype == "word":
                        dist += 1
                    elif mtype == "ref":
                        if cat in self.ref_cats:
                            dist += 1
                if curr_id == line_id:
                    #assert dist > 0 #else cataphor. could run over reversed lines to get cataphors, but i think only one or two are in the corpus?
                    #TODO fix for cataphors
                    return dist
                
    def get_dist_pro(self, mtype, curr_id, ante_id):
        #TODO
        pass

    def get_chain_size(self, curr_id, ante_id):
        if ante_id not in self.chainsize:
            self.chainsize[curr_id] = 1
        else:
            self.chainsize[curr_id] = self.chainsize[ante_id] + 1
        return self.chainsize[curr_id]
   
    def get_chain_size_pro(self, curr_id, ante_id):
        #TODO
        #lookup chain_size_pro.  if exists, use ante_id, add curr with increment. if does not exist, recurse with ante_id of ante_id.
        if ante_id in self.chain_size_pro:
            word = self.id2word[curr_id]
            if self.is_pro(word):
                self.chain_size_pro[curr_id] = self.chain_size_pro[ante_id] + 1
            return self.chain_size_pro[ante_id]

        #base case - hit-non-anaphor id for ante_id. set chain_size_pro[curr_id] = size 1, return chain_size_pro[curr_id]
        
        pass

    def get_story_pos(self, sentidx):
        #find story given sentidx
        storyid = self.sentidx2storyid(sentidx)
        #lookup story start idx
        storystartidx = self.storyidxes[storyid][0]
        #lookup story length in sentences
        storylen = self.storyidxes[storyid][1] - self.storyidxes[storyid][0]
        #convert sentidx to be 1-indexed (subtract start index of story)
        csentidx = sentidx - storystartidx
        #return story pos as 1indexed sentidx / storylen
        storypos = csentidx / storylen
        return storypos

    def get_predictors(self):
        for idx, line in enumerate(self.lines):
            #if idx % 100 == 0:
                #print("Processing line {0}".format(idx))
            #if idx % 100 == 0 and idx > 1: #testing only - REMOVE THESE TWO LINES
            #    break
            curr_id = line.split(":")[0] #current word id group
            #last2 are wordidx, rest are sentidx
            sentidx = int(curr_id[:-2])
            story_pos = self.get_story_pos(sentidx)
            match = re.search(r".*-[nm]([0-9]+) ", line)
            if match is not None:
                ante_id = match.group(1) # antecedent id group
                binary_coref_indic = 1
                word_dist = self.get_dist("word", curr_id, ante_id)
                ref_dist = self.get_dist("ref", curr_id, ante_id)
                chain_size = self.get_chain_size(curr_id, ante_id)

                #self.predictors.append([curr_id, str(binary_coref_indic), str(word_dist), str(ref_dist), str(chain_size), str(ispro)]) #list of lists
                word = self.get_word(line)
                if self.is_pro(word): #don't need to check category because already restricted to anaphoric instances. i.e., already excludes relativizers, determiners, etc.
                    isanaphpro = "y"
                else:
                    isanaphpro = "n"

            else:
                binary_coref_indic = 0
                word_dist = "nan"
                ref_dist = "nan"
                chain_size = "nan"
                isanaphpro = "n"

                #self.predictors.append([curr_id, str(binary_coref_indic),"0","0","0","0"])
                    
            try:
                word_dist_log = log1p(word_dist)
                ref_dist_log = log1p(ref_dist)
                chain_size_log = log1p(chain_size)
            except TypeError:
                word_dist_log = "nan"
                ref_dist_log = "nan"
                chain_size_log = "nan"
            #self.predictors.append([curr_id, str(binary_coref_indic), str(word_dist), str(ref_dist), str(chain_size), isanaphpro]) #list of lists
            self.predictors.append([curr_id, str(binary_coref_indic), str(word_dist), str(ref_dist), str(chain_size), isanaphpro, str(word_dist_log), str(ref_dist_log), str(chain_size_log), str(story_pos)]) #list of lists
                

    def print_predictors(self):
        #pdb.set_trace()
        print(DELIM.join(["uid", "corefbin", "coreflenw", "coreflenr", "corefsize", "isanaphpro", "coreflenwlog", "coreflenrlog", "corefsizelog", "storypos"]))
        for p in self.predictors:
            print(DELIM.join(p))

def main():
    #print("Beginning feature extraction...")
    with open(sys.argv[1], 'r') as iff:
        lines = iff.readlines()

    cp = CorefPredictors(lines)
    cp.get_predictors()
    cp.print_predictors()

if __name__ == "__main__":
    main()
