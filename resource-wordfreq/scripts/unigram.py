import sys
import math
from wordfreq import word_frequency

print("word unigramsurp")

lang_dict = {"du": "nl",
             "ge": "de",
             "gr": "el",
             "no": "nb",
             "sp": "es"}

lang = sys.argv[1]
if lang in lang_dict:
    lang = lang_dict[lang]

with open(sys.argv[2], "r") as f:
    for line in f:
        if line.strip() == "!ARTICLE":
            continue
        for word in line.strip().split():
            # uni_surp = -1 * math.log2(word_frequency(word, lang, wordlist='best', minimum=0.0))
            uni_surp = -1 * math.log2(word_frequency(word, lang, wordlist='best', minimum=1e-10))
            print(word + ' ' + str(uni_surp))