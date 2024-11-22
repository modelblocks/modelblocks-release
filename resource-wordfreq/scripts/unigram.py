import sys
import math
from wordfreq import word_frequency

print("word unigramsurp")

lang = sys.argv[1]
with open(sys.argv[2], "r") as f:
    for line in f:
        for word in line.strip().split():
            # uni_surp = -1 * math.log2(word_frequency(word, lang, wordlist='best', minimum=0.0))
            uni_surp = -1 * math.log2(word_frequency(word, lang, wordlist='best', minimum=1e-10))
            print(word + ' ' + str(uni_surp))
