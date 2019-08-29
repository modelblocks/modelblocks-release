'''
Title: extract_coref_predictors.py
Author: Evan Jaffe
Original date: 3/28/2017

Generate by-word tsv of predictors for coreference-based features. Uses annotated numbered editable linetrees file, with X-nID and X-mID as indicating the id of the antecedent.

To run:
    python extract_coref_predictors.py <input_filename>

"uid", "word", "corefbin", "coreflenw", "coreflenr", "corefsize", "isanaphpro", "coreflenwlog", "coreflenrlog", "corefsizelog", "storypos", "storypos13bin", "storypos23bin", "storypos33bin", "storypos12bin", "storypos22bin", "maxcorefsize"

Output format:
Column 1: word_id, in sentenceid_wordid format
Column 2: word, as string
Column 3: binary corefence indicator
Column 4: distance in words to antecedent
Column 5: distance in nouns,verbs to antecedent
Column 6: size of chain in mention count up to current mention
Column 7: binary indicator if is anaphoric and pronominal
Column 8: distance in words, log transformed
Column 9: distance in referents, log transformed
Column 10: incremental mention count size, log transformed
Column 11: relative sentence position in story, from 0 to 1
Column 12: binary indicator if sentence in 1st third of story
Column 13: binary indicator if sentence in 2nd third of story
Column 14: binary indicator if sentence in 3rd third of story
Column 15: binary indicator if sentence in 1st half of story
Column 16: binary indicator if sentence in 2nd half of story
Column 17: size of chain in mention count over entire story - fixed max for every mention
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
        self.chains = []
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
                
    '''
    def get_dist_pro(self, mtype, curr_id, ante_id):
        #TODO
        pass
    '''

    def get_chain_size(self, curr_id, ante_id):
        if ante_id not in self.chainsize:
            self.chainsize[curr_id] = 1
        else:
            self.chainsize[curr_id] = self.chainsize[ante_id] + 1
        return self.chainsize[curr_id]
   
    '''
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
    '''

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

    def get_story_pos_bins(self, storypos):
        assert storypos <= 1 and  storypos >=0
        first3rd = "y" if storypos < 0.333 else "n"
        second3rd = "y" if (storypos >= 0.333 and storypos <= 0.666) else "n"
        third3rd = "y" if storypos > 0.666 else "n"
        firsthalf = "y" if storypos < 0.5 else "n"
        secondhalf = "y" if storypos >= 0.5 else "n"
        return first3rd, second3rd, third3rd, firsthalf, secondhalf

    def find_id_in_chains(self, id):
        for idx, chain in enumerate(self.chains):
            if id in chain:
                return idx
        return None

    def accumulate_chains(self):
        for idx, line in enumerate(self.lines):
            curr_id = line.split(":")[0] #current word id group
            #last2 are wordidx, rest are sentidx
            sentidx = int(curr_id[:-2])
            match = re.search(r".*-[nm]([0-9]+) ", line)
            if match is not None:
                ante_id = match.group(1) # antecedent id group
                # check if ante_id already exists in some chain 
                chain_idx = self.find_id_in_chains(ante_id) #
                # if it doesn't, create new chain [ante_id, curr_id]
                if chain_idx == None:
                    self.chains.append([ante_id, curr_id])
                # if does, append curr_id to that chain
                else:
                    self.chains[chain_idx] += [curr_id]

    def get_max_chain_size(self, id):
        idx = self.find_id_in_chains(id)
        if idx == None:
            return "nan"
        else:
            return len(self.chains[idx])-1 #singleton/one mention has zero chain size, one repeat = chainsize 1, etc.
        
        
    def get_predictors(self):
        self.accumulate_chains()
        #pdb.set_trace()
        for idx, line in enumerate(self.lines):
            #if idx % 100 == 0:
                #print("Processing line {0}".format(idx))
            #if idx % 100 == 0 and idx > 1: #testing only - REMOVE THESE TWO LINES
            #    break
            curr_id = line.split(":")[0] #current word id group
            #last2 are wordidx, rest are sentidx
            sentidx = int(curr_id[:-2])
            story_pos = self.get_story_pos(sentidx)
            storypos13bin, storypos23bin, storypos33bin, storypos12bin, storypos22bin  = self.get_story_pos_bins(story_pos)
            max_chain_size = self.get_max_chain_size(curr_id)
            word = self.get_word(line)

            #check for anaphoricity
            match = re.search(r".*-[nm]([0-9]+) ", line)
            if match is not None:
                ante_id = match.group(1) # antecedent id group
                binary_coref_indic = 1
                word_dist = self.get_dist("word", curr_id, ante_id)
                ref_dist = self.get_dist("ref", curr_id, ante_id)
                chain_size = self.get_chain_size(curr_id, ante_id)
                #self.predictors.append([curr_id, str(binary_coref_indic), str(word_dist), str(ref_dist), str(chain_size), str(ispro)]) #list of lists
            
                isanaphpro = "y" if self.is_pro(word) else "n" #don't need to check category because already restricted to anaphoric instances. i.e., already excludes relativizers, determiners, etc.

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
            self.predictors.append([curr_id, str(word), str(binary_coref_indic), str(word_dist), str(ref_dist), str(chain_size), isanaphpro, str(word_dist_log), str(ref_dist_log), str(chain_size_log), str(story_pos), str(storypos13bin), str(storypos23bin), str(storypos33bin), str(storypos12bin), str(storypos22bin), str(max_chain_size)]) #list of lists
                

    def print_predictors(self):
        #pdb.set_trace()
        print(DELIM.join(["uid", "word", "corefbin", "coreflenw", "coreflenr", "corefsize", "isanaphpro", "coreflenwlog", "coreflenrlog", "corefsizelog", "storypos", "storypos13bin", "storypos23bin", "storypos33bin", "storypos12bin", "storypos22bin", "maxcorefsize"]))
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
