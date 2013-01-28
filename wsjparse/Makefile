###############################################################################
##                                                                           ##
## This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                           ##
##    ModelBlocks is free software: you can redistribute it and/or modify    ##
##    it under the terms of the GNU General Public License as published by   ##
##    the Free Software Foundation, either version 3 of the License, or      ##
##    (at your option) any later version.                                    ##
##                                                                           ##
##    ModelBlocks is distributed in the hope that it will be useful,         ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of         ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          ##
##    GNU General Public License for more details.                           ##
##                                                                           ##
##    You should have received a copy of the GNU General Public License      ##
##    along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.   ##
##                                                                           ##
###############################################################################

################################################################################
#
#  i. Macros & variables
#
################################################################################

SHELL = /bin/bash
INCLUDES = -Iinclude -I../rvtl/include -I../slush/include #-I/sw/include #-I/Users/dingcheng/Documents/boost/boost_1_44_0
CFLAGS = $(INCLUDES) -Wall `cat user-cflags.txt` -g #-DNDEBUG -O3 #-DNOWARNINGS #-g #
CC = g++ 
LD = g++
PYTHON = python3
ME_PARAMS = -i500,-g0.5
#ME_PARAMS = -i100
#X_MXSTUFF = 
X_MXSTUFF = -Xmx12g

comma = ,
space = $(subst s, ,s)

PROPTXT       = /project/nlp/data/propbank/propbank-1.0/prop.txt
READINGDATA   = /project/nlp/data/readingtimes

WSJMAPTRAINSECTS  = 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21  ##EOS
WSJTRAINSECTS  = 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21  ##EOS
WSJTRAINSECTS02  = 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21  ##EOS
WSJHELDOUTSECTS  = 00
BROWNTRAINSECTS = cf cg ck cl cm cn cp cr
LRSECTS = oerr oer ofr oq rnr se ser

include $(wildcard */*.d)      ## don't comment out, it breaks make!

.SUFFIXES:
.SECONDEXPANSION:

################################################################################
#
#  ii. Published Results
#
################################################################################

## reannotate/unbounded-dependency paper (nguyen et al 2012)
pub: lrtest.wsj02to21-gcg-1671-3sm.berk.parsed.gapeval

################################################################################
#
#  iii. Demo
#
################################################################################

## std model, berk parser
all: wsj22-10first.wsj01to21-gcg-nol-1671-3sm.berk.parsed.cnftrees
## std model, cpt parser (deprecated)
all: wsj22-10first.wsj01to21-gcg-nol-1671-3sm.unked.wsj01to21-gcg-nol-1671-3sm-bd.x-cfp.-xa_parsed.nosm.cnftrees

################################################################################
#
#  iv. User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

#### c++ compile flags
user-cflags.txt:   ## -g
	echo '-DNDEBUG -O3' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may not be to your liking'
	@echo 'edit it to tell C++ whether to compile in debug mode or optimize, and re-run make to continue!'
	@echo ''

#### c++ compile flags
user-javaflags.txt:
	echo '-Xmx4g' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may not be to your liking'
	@echo 'edit it to give java as much memory as you want to, and re-run make to continue!'
	@echo ''

#### location of treebank
user-treebank-location.txt:
	echo '/home/corpora/original/english/penn_treebank_3' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your treebank repository, and re-run make to continue!'
	@echo ''

#### location of lrbank
user-lrbank-location.txt:
	echo '/home/corpora/original/english/longrange' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your lrbank repository, and re-run make to continue!'
	@echo ''

#### location of propbank
user-propbank-location.txt:
	echo '/home/corpora/original/english/propbank/propbank-1.0' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your propbank repository, and re-run make to continue!'
	@echo ''

#### includes for user sub-projects
include user-subproject-includes.txt
user-subproject-includes.txt:
	echo '' > $@


################################################################################
#
#  v. Code compilation items
#
################################################################################

#### bin directory (ignored by git b/c empty)
bin:
	if [ ! -d $@ ]; then mkdir $@; fi


#### c++ dependencies
.PRECIOUS: %.d
%.d: %.cpp
	echo '$*.d: \' > $@   #' ##
	echo `$(CC) $(INCLUDES) -MM $<` | sed 's/^.*:.*\.cpp */ $$(wildcard /;s/\\ *//g;s/$$/)/' >> $@
	cat $@ | sed 's/\.d:/\.cpp:/' >> $@
#### ocaml dependencies
%.d: %.ml
	echo '$@ : \' > $@   #' ##
	echo `grep '#load' $< | sed 's/.*\"\(.*\)\".*/\1/' | grep -v 'cmxa'` | sed 's/\.ml/\.cmx/g' >> $@
	echo '$< : \' >> $@   #' ##
	echo `grep '#load' $< | sed 's/.*\"\(.*\)\".*/\1/' | grep -v 'cmxa'` | sed 's/\.ml/\.cmx/g' >> $@


#### c++ executables
.PRECIOUS: bin/%
bin/%: src/%.cpp src/%.d user-cflags.txt | bin
	$(CC) $(CFLAGS) -lm $< -o $@
#### ocaml executables
%.cmx: %.ml %.d
	ocamlopt -I scripts `grep '#load' $< | sed 's/.*\"\(.*\)\".*/\1/' | sed 's/\.ml/\.cmx/g'` $< -c $@
bin/%: scripts/%.ml scripts/%.d | bin
	ocamlopt -I scripts `grep '#load' $< | sed 's/.*\"\(.*\)\".*/\1/' | sed 's/\.ml/\.cmx/g'` $< -o $@
#### cython executables
%.c: %.py
	cython --embed $< -o $@
bin/%: scripts/%.c
	gcc  -lpython2.5 -I /Library/Frameworks/Python.framework/Versions/2.6/include/python2.6/ $< -o $@
#### java executable objects
.PRECIOUS: %.class
%.class: %.java #$$(addsuffix .class,$$(subst .,/,$$(subst import ,,$$(shell grep -o 'import edu[.a-zA-Z0-9]*' $$(subst .class,.java,$$@)))))
	javac $<

#### maxent executable
../maxent-20061005/src/opt/maxent: ../liblbfgs-1.10/lib/lbfgs.o 
	cd ../maxent-20061005 ; ./configure ; make
../liblbfgs-1.10/lib/lbfgs.o:
	cd ../liblbfgs-1.10 ; ./configure ; make

#### berkeley parser shortcut "executable"
bin/parser-berk:  edu/berkeley/nlp/PCFGLA/BerkeleyParser.class  user-javaflags.txt  |  bin
	echo "#!/bin/sh" > $@
	echo "java $(shell cat $(word 2,$^)) edu/berkeley/nlp/PCFGLA/BerkeleyParser -viterbi -substates -binarize -gr \$$1" >> $@
	chmod u+x $@

bin/calc-cfp-hhmm: src/calc-cfp-hhmm.c src/model.c
	gcc -Wall -Isrc -o $@ $^

#### pdf formatted printouts for electronic markup
%.ps: %
	cat $< | enscript -fCourier7 -r -o $@
%.ln.ps: %
	cat $< | grep -n '' | enscript -fCourier7 -r -o $@
%.pdf: %.ps
	ps2pdf $< $@


#### GPL packages from other authors included in modelblocks
bin/evalb: src/evalb.c | bin
	gcc -Wall -g -o $@ $<
stanford-tools.jar: # for the parser and for tree surgeon
	javac ../edu/stanford/nlp/*/*.java ../edu/stanford/nlp/*/*/*.java ../edu/stanford/nlp/*/*/*/*.java
	cd ../ ;  jar -cf wsjparse/stanford-tools.jar edu/ ; cd wsjparse
berkeley-parser:
	if [ ! -d $@ ]; then mkdir $@; fi
berkeley-parser/berkeleyParser.jar: | berkeley-parser
	javac edu/berkeley/nlp/ling/CollinsHeadFinder.java
	javac edu/berkeley/nlp/PCFGLA/BerkeleyParser.java 
	javac fig/basic/Pair.java
	javac edu/berkeley/nlp/PCFGLA/WriteGrammarToTextFile.java
	jar -cf berkeley-parser/berkeleyParser.jar edu/ fig/



################################################################################
#
#  1. Text formatting items
#
################################################################################

#### obtain txt from html by recursively removing matched pairs of markup (except for those containing the article), then unpaired markup
%.txt: %.html
	cat $< | tr -d '\n' | perl -pe 's/^.*<body[^>]*>(.*)<.body>.*$$/\1/g;  s/<p>/ /g;  s/<(a) ?.*?>(.*?)<\/\1>/\2/g;  s/<(b|i) ?.*?>(.*?)<\/\1>/\2/g;  while ( s/<(?!div id="bodyContent")([a-z0-9]*)[ >](.(?!<\1))*?<\/\1>//g ){};  s/<[^>]*>//g;  s/^[ \t]*(.*?)[ \t\n]*$$/\1/g;' | perl -pe "s/&#160;/ /g;  s/—/--/g" > $@

%.sents: %.txt
	cat $< | perl -pe "s/\"(?! )(?!\))/\`\`/g;  s/(?<! )\"/''/g" | perl -pe "s/(?<! c)(?<! ca)([\.!?])(\)|'')? */ \1 \2\n/g;  s/(\(|\`\`)(?! )/\1 /g;  s/(?<! )(\)|,|;|:|'')(?![0-9])/ \1/g;  y/[A-Z]/[a-z]/"  |  perl -pe "s/\(/\!lrb\!/g;  s/\)/\!rrb\!/g;  s/;/\!semi\!/g;  s/:/\!colon\!/g"  >  $@

#### obtain lexicon of terms that appear more than 5 times
.PRECIOUS: %.lexicon
%.lexicon: %.sents scripts/buildLexicon.py
	$(PYTHON) $(word 2,$^) $< -u5 > $@

################################################################################
#
#  2. Syntax formatting items
#
#  to construct the following file types:
#    <x>.linetrees    : treebank-tagset phase structure trees, one sentence per line, bracketed by parens
#    <x>.cnftrees     : fully-binarized chomsky normal form phase structure trees, one sentence per line, bracketed by parens
#    <x>.gaptrees     : gap-annotated phase structure trees, one sentence per line, bracketed by parens
#                       tagset augmented with: -g (constituent contains unfilled gap)
#                                              -m (constituent marked with punctuation:
#                                                  -mB=brack/paren, -mC=comma, -mD=dash, -mE=eos punct, -mS=semi)
#                                              -n (constituent followed by punctuation: [as above])
#                                              -p (constituent preceded by punctuation: [as above])
#    <x>-mb.cnftrees  : fully-binarized chomsky normal form phase structure trees, one sentence per line, bracketed by parens
#    <x>-nr.cnftrees  : <x>.cnftrees, with all 'rare' (single-token type) categories replaced with 'unk'
#    <x>-rl.cnftrees  : role-labeled chomsky normal form phase structure trees, one sentence per line, bracketed by parens
#                       tagset augmented with: -l (constituent marked with role label, describing relation to parent:
#                                                  -lA=argument, -lM=modifier, -lI=identity, -lC=conjunct, -lN=no relation)
#
################################################################################

#### genmodel directory (ignored by git b/c empty)
genmodel:
	if [ ! -d $@ ]; then mkdir $@; fi


#### obtain sentence-aligned tree files (with one tree per line), extracted from various treebanks
## for wsj corpus
.PRECIOUS: genmodel/wsj%.linetrees     
genmodel/wsj%.linetrees:   user-treebank-location.txt $$(shell cat user-treebank-location.txt)/parsed/mrg/wsj/% scripts/tbtrees2linetrees.pl | genmodel
	cat $(word 2,$^)/*.mrg | perl $(word 3,$^) > $@
## for brown corpus
.PRECIOUS: genmodel/brown%.linetrees
genmodel/brown%.linetrees: user-treebank-location.txt $$(shell cat user-treebank-location.txt)/parsed/mrg/brown/% scripts/tbtrees2linetrees.pl
	cat $(word 2,$^)/*.mrg | perl $(word 3,$^) > $@

## for unbounded dependency corpus
.PRECIOUS: genmodel/lr%.sents
.PRECIOUS: genmodel/lr%.ans

genmodel/lr%oerr.sents: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_extract_red_rel/%.raw.obj_extract_red_rel
	cat $(word 2,$^) | perl -pe "s/\(/-RRB-/g;s/\)/-LRB-/g;" > $@
genmodel/lr%oerr.ans: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_extract_red_rel/%.obj_extract_red_rel
	cp $(word 2,$^) $@
genmodel/lr%oer.sents: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_extract_rel_clause/%.raw.obj_extract_rel_clause
	cat $(word 2,$^) | perl -pe "s/\(/-RRB-/g;s/\)/-LRB-/g;" > $@
genmodel/lr%oer.ans: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_extract_rel_clause/%.obj_extract_rel_clause
	cp $(word 2,$^) $@
genmodel/lr%ofr.sents: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_free_rels/%.raw.obj_free_rels
	cat $(word 2,$^) | perl -pe "s/\(/-RRB-/g;s/\)/-LRB-/g;" > $@
genmodel/lr%ofr.ans: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_free_rels/%.obj_free_rels
	cp $(word 2,$^) $@
genmodel/lr%oq.sents: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_qus/%.raw.obj_qus
	cat $(word 2,$^) | perl -pe "s/\(/-RRB-/g;s/\)/-LRB-/g;" > $@
genmodel/lr%oq.ans: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/obj_qus/%.obj_qus
	cp $(word 2,$^) $@
genmodel/lr%rnr.sents: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/right_node_raising/%.raw.right_node_raising
	cat $(word 2,$^) | perl -pe "s/\(/-RRB-/g;s/\)/-LRB-/g;" > $@
genmodel/lr%rnr.ans: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/right_node_raising/%.right_node_raising
	cp $(word 2,$^) $@
genmodel/lr%se.sents: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/sbj_embedded/%.raw.sbj_embedded
	cat $(word 2,$^) | perl -pe "s/\(/-RRB-/g;s/\)/-LRB-/g;" > $@
genmodel/lr%se.ans: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/sbj_embedded/%.sbj_embedded
	cp $(word 2,$^) $@
genmodel/lr%ser.sents: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/sbj_extract_rel_clause/%.raw.sbj_extract_rel_clause
	cat $(word 2,$^) | perl -pe "s/\(/-RRB-/g;s/\)/-LRB-/g;" > $@
genmodel/lr%ser.ans: user-lrbank-location.txt $$(shell cat user-lrbank-location.txt)/sbj_extract_rel_clause/%.sbj_extract_rel_clause
	cp $(word 2,$^) $@

lrdev.%:  lrdevoerr.%  lrdevoer.%  lrdevofr.%  lrdevoq.%  lrdevrnr.%  lrdevse.%  lrdevser.%
	more $^  >  $@
lrtest.%: lrtestoerr.% lrtestoer.% lrtestofr.% lrtestoq.% lrtestrnr.% lrtestse.% lrtestser.%
	more $^  >  $@

#### collections
genmodel/wsj00to21.linetrees: $(foreach sect,$(WSJMAPTRAINSECTS),genmodel/wsj$(sect).linetrees)
	cat $^  >  $@
genmodel/wsj01to21.linetrees: $(foreach sect,$(WSJTRAINSECTS),genmodel/wsj$(sect).linetrees)
	cat $^  >  $@
genmodel/wsj02to21.linetrees: $(foreach sect,$(WSJTRAINSECTS02),genmodel/wsj$(sect).linetrees)
	cat $^  >  $@

genmodel/brownTRAIN.linetrees: $(foreach sect,$(BROWNTRAINSECTS),genmodel/brown$(sect).linetrees)
	cat $^ > $@

#### selective tree sets: look for selections in the order they occur below:
#### use only trees with N or greater words
%.linetrees: $$(wordlist 2,$$(words $$(subst -minwds, ,$$@)),- $$(subst -minwds, ,$$@)).linetrees
	cat $< | perl -na -e "if (split(/\#/)>$(word 2,$(subst -minwds, ,$*))) {print $$_;}" > $@      ## (note: split >=X would mean num words >=X-1)
#### use only first N trees
%.linetrees: $$(wordlist 2,$$(words $$(subst -first, ,$$@)),- $$(subst -first, ,$$@)).linetrees
	head -$(word 2,$(subst -first, ,$*)) $< > $@
%first.linetrees: $$(subst $$(space),-,$$(wordlist 2,$$(words $$(subst -, ,$$@)),- $$(subst -, ,$$@))).linetrees
	head -$(lastword $(subst -, ,$*)) $< > $@
%last.linetrees: $$(subst $$(space),-,$$(wordlist 2,$$(words $$(subst -, ,$$@)),- $$(subst -, ,$$@))).linetrees
	tail -$(lastword $(subst -, ,$*)) $< > $@

%.linetrees: $$(wordlist 2,$$(words $$(subst -only, ,$$@)),- $$(subst -only, ,$$@)).linetrees
	head -$(word 2,$(subst -first, ,$*)) $< > $@
%only.linetrees: $$(subst $$(space),-,$$(wordlist 2,$$(words $$(subst -, ,$$@)),- $$(subst -, ,$$@))).linetrees
	head -$(lastword $(subst -, ,$*)) $< | tail -1 > $@

%first.linetrees:  $$(basename %).linetrees
	head -$(subst .,,$(suffix $*)) $<  >  $@
%last.linetrees:  $$(basename %).linetrees
	tail -$(subst .,,$(suffix $*)) $<  >  $@
%only.linetrees:  $$(basename %).linetrees
	head -$(subst .,,$(suffix $*)) $< | tail -1 >  $@
%all.linetrees: $$(basename %).linetrees
	cat $< > $@
%-all.linetrees: $$(basename %).linetrees
	cat $< > $@

%.nounary.linetrees:  %.linetrees  scripts/killUnaries.pl
	cat $<  |  perl $(word 2,$^)  >  $@

.PRECIOUS: %.gcg.cnftrees  ## NOTE: inline seds dispose of reannotation failures, then remove -f tags, then shift -l tags to end
%.gcg.cnftrees:  %.linetrees  scripts/annotateFixes.pl  scripts/annotateGcg.pl  scripts/killUnaries.pl
	cat $<  |  perl $(word 2,$^) -p |  perl $(word 3,$^)  |  perl $(word 4,$^) -l  |  perl -pe 's/.*\([A-Z]+ .*//'  |  perl -pe 's/\(([^ ]*)-f[^ ]*/\(\1/g' | perl -pe 's/\(([^ ]*)-l([A-Z])([^ ]*)/\(\1\3-l\2/g;'  >  $@

.PRECIOUS: %.gcg.hy.cnftrees ## this breaks hyphenated and slashed words for conll08 eval
%.gcg.hy.cnftrees:  %.gcg.cnftrees scripts/dehyphenate.py scripts/conll08Common.py
	python3 $(word 2,$^) -c $< -d1 >  $@

%first.cnftrees:  $$(basename %).cnftrees
	head -$(subst .,,$(suffix $*)) $<  >  $@
%last.cnftrees:  $$(basename %).cnftrees
	tail -$(subst .,,$(suffix $*)) $<  >  $@
%only.cnftrees:  $$(basename %).cnftrees
	head -$(subst .,,$(suffix $*)) $< | tail -1 >  $@
%.extrpar.cnftrees:  %.cnftrees
	cat $<  |  perl -pe 's/\-(?![^ \)]*\))/\&/g'  |  sed 's/^\((.*\)$$/(\1)/g'  >  $@
%.extrpar.linetrees: %.linetrees
	cat $<  |  sed 's/^\((.*\)$$/(\1)/g'  >  $@
%.bd.cnftrees:  %.cnftrees  scripts/annotateDepth.py
	cat $<  |  python3 $(word 2,$^)  >  $@  # |  grep -v '\-bR-d5'  >  $@
%.nobd.cnftrees:  %.cnftrees
	cat $<  |  sed 's/-b[LR]-d[0-9]//g'  >  $@
%.nol.cnftrees:  %.cnftrees
	cat $<  |  sed 's/-l.//g' > $@
%.fg.cnftrees:  %.cnftrees  scripts/annotateFGTrans.pl
	cat $<  |  perl $(word 2,$^)  >  $@

#### symbol counts
%.symbolcounts: %
	cat $< | bin/indent2 | sed 's/^ *(\([^ ]*\).*$$/\1/' | sort | uniq -c > $@

%.viewtrees: %.cnftrees scripts/viewtree
	$(word 2,$^) $<

################################################################################
#
#  3. Propositional content formatting items
#
#  to construct the following file types:
#    <x>.sentrelns : sentence-indexed relations, one per line, with role-specific propositions delimited by spaces
#    <x>.pbconts   : propbank-tagset sentence contents, one sentence per line, with role-specific propositions delimited by spaces
#    <x>.melconts  : melcuk-tagset sentence contents, one sentence per line, with role-specific propositions delimited by spaces
#
################################################################################

#### obtain relation-aligned sentence-indexed propositions for each sentence, including empty lines for `proposition-free' sentences
genmodel/wsj%.sentrelns:  user-treebank-location.txt  $$(shell cat user-treebank-location.txt)/parsed/mrg/wsj/%  user-propbank-location.txt  $$(shell cat user-propbank-location.txt)/prop.txt 
	grep '^(' $(word 2,$^)/*.mrg  |  sed 's/.*parsed\/mrg\///;s/:.*//'  |  perl -pe 'if($$prev ne $$_){$$prev=$$_;$$ct=0;} s/$$/ $$ct/; $$ct++;'  >  $@
	cat $(word 4,$^)  |  grep '^wsj.$*'  >>  $@
#### obtain relation-aligned sentence-indexed propositions for entire training set, including empty lines for `proposition-free' sentences
genmodel/wsj00to21.sentrelns: $(foreach sect,$(WSJMAPTRAINSECTS),genmodel/wsj$(sect).sentrelns) ##genmodel/wsjEOS$*trees  ##genmodel/eos.cnftrees
	cat $^ > $@
genmodel/wsj01to21.sentrelns: $(foreach sect,$(WSJTRAINSECTS),genmodel/wsj$(sect).sentrelns) ##genmodel/wsjEOS$*trees  ##genmodel/eos.cnftrees
	cat $^ > $@
genmodel/wsj02to21.sentrelns: $(foreach sect,$(WSJTRAINSECTS02),genmodel/wsj$(sect).sentrelns) ##genmodel/wsjEOS$*trees  ##genmodel/eos.cnftrees
	cat $^ > $@
genmodel/wsjTRAIN.sentrelns: $(foreach sect,$(WSJTRAINSECTS),genmodel/wsj$(sect).sentrelns) ##genmodel/wsjEOS$*trees  ##genmodel/eos.cnftrees
	cat $^ > $@


#### obtain sentence-aligned space-delimited propbank-domain propositions
.PRECIOUS: %.pbconts
%.pbconts: %.sentrelns  scripts/sentrelns2pbconts.py  %.linetrees
	cat $<  |  perl -pe 'while(s/([^ ]*)\*([^-]*)-([^ \n]*)/\1-\3 \2-\3/g){}; while(s/([^ ]*),([^-]*)-([^ \n]*)/\1-\3 \2-\3/g){}'  |  $(PYTHON) $(word 2,$^) $(word 3,$^)  >  $@
#### use only first N trees
%.pbconts: $$(wordlist 2,$$(words $$(subst -first, ,$$@)),- $$(subst -first, ,$$@)).pbconts
	head -$(word 2,$(subst -first, ,$*)) $< > $@
%first.pbconts: $$(subst $$(space),-,$$(wordlist 2,$$(words $$(subst -, ,$$@)),- $$(subst -, ,$$@))).pbconts
	head -$(lastword $(subst -, ,$*)) $< > $@


#### obtain sentence-aligned space-delimited text-based(number)-domain propositions
.PRECIOUS: %.tbconts
%berk.parsed.tbconts:  %berk.parsed.cnftrees scripts/trees2melconts.py
	cat $< | sed 's/(\(\))/\[\)/g;' | perl -pe 's/(\([^ ]+) +\)/\1 \]/g;' > $@.tmp
	$(PYTHON) $(word 2,$^) -t $@.tmp -p -r -c > $@
	rm -f $@.tmp

%_parsed.tbconts:  %_parsed.output scripts/output2tbconts.py scripts/output2commonconts.py
	$(PYTHON) $(word 2,$^) $<  >  $@
%.ans.tbconts:  %.ans  scripts/convertGoldUnbound.py
	$(PYTHON) $(word 2,$^) -f $< >  $@


#### obtain sentence-aligned space-delimited melcuk(number)-domain propositions
.PRECIOUS: %.melconts
%.melconts:  %.cnftrees  scripts/trees2melconts.py
	$(PYTHON) $(word 2,$^) -t $< -p -r >  $@
%.ans.melconts:  %.ans  scripts/convertGoldUnbound.py
	$(PYTHON) $(word 2,$^) -f $< >  $@
%nosm.melconts:  %nosm.cnftrees  scripts/trees2melconts.py
	$(PYTHON) $(word 2,$^) -t $< -p >  $@
%parsed.melconts:  %parsed.output scripts/output2melconts.py scripts/output2commonconts.py
	$(PYTHON) $(word 2,$^) $<  >  $@
%.enju.melconts:  %.enju.output scripts/enju2melconts.py
	$(PYTHON) $(word 2,$^) $<  >  $@

%.melrels:  %.melconts
	cat $<  |  grep -n ''  |  perl -pe 's/^([0-9]+):/-----\1-----\n/;s/ (?!$$)/\n/g'  >  $@

################################################################################
#
#  4. Syntax model building items
#
#  to construct the following file types:
#
#    <m>.<x>-<y>.counts : raw counts of model patterns in <m>, using observed <x> and hidden <y> random variables:
#
#                                = x    : X    (likelihood of word given terminal category)
#
#                         if <y> = cc   : Cr   (root prior probabilities)
#                                         CC   (joint left and right child given parent)
#
#                                = ccu  : Cr,CC (as above)
#                                         Cu   (unary child given parent --- not used in cnf parsing)
#
#                                = cfp  : Ce   (expansion probability)
#                                         Ctaa (active component of active transition)
#                                         Ctaw (awaited component of active transition)
#                                         Ctww (awaited component of awaited transition)
#                                         F    (final state flag)
#
#    <m>.<x>-<y>.model : probability model, based on counts
#
################################################################################

#### x-cc counts
.PRECIOUS: %.x-cc.counts
%.x-cc.counts: %.cnftrees  scripts/trees2rules.pl  scripts/relfreq.pl
	cat $<  |  perl $(word 2,$^)  |  perl $(word 3,$^) -f  >  $@

####
.PRECIOUS: %.x.model
%.model:  %.counts  scripts/relfreq.pl
	cat $<  |  perl $(word 2,$^)  >  $@

#### berkeley model -- obtain grammar and lexicon (text files)
.PRECIOUS: %.splits %.grammar %.lexicon
%.splits %.grammar %.lexicon:  berkeley-parser/berkeleyParser.jar  %.gr  user-javaflags.txt
	java  $(shell cat $(word 3,$^))  -cp $<  edu/berkeley/nlp/PCFGLA/WriteGrammarToTextFile  $(word 2,$^)  $(basename $@)
.PRECIOUS: %.x-ccu.model
#### berkeley model -- obtain x-cc model (with unaries)
%.x-ccu.model:  %.grammar  %.lexicon  scripts/berkgrammar2ckygr.py  scripts/berklexicon2ckylex.py
	cat $(word 1,$^)  |  python3 $(word 3,$^)  >   $@
	cat $(word 2,$^)  |  python3 $(word 4,$^)  >>  $@

####
.PRECIOUS: %sm.gr
%sm.gr:  edu/berkeley/nlp/PCFGLA/GrammarTrainer.class  $$(basename $$(basename %)).extrpar.cnftrees  $$(basename %)last.extrpar.cnftrees  user-javaflags.txt
	java  $(shell cat $(word 4,$^))  $(subst /,.,$(basename $<))  -SMcycles $(subst .,,$(suffix $*))  -path $(word 2,$^)  -validation $(word 3,$^)  -treebank SINGLEFILE  -out $@

%sm.gr:  edu/berkeley/nlp/PCFGLA/GrammarTrainer.class  $$(basename $$(basename %)).extrpar.linetrees  $$(basename %)last.extrpar.linetrees  user-javaflags.txt
	java  $(shell cat $(word 4,$^))  $(subst /,.,$(basename $<))  -SMcycles $(subst .,,$(suffix $*))  -path $(word 2,$^)  -validation $(word 3,$^)  -treebank SINGLEFILE  -out $@


#### silly shortcut for berkeley parser
%.berk.model: %.gr
	ln -sf $(notdir $<) $@

#### obtain strict cc model (no unaries)
.PRECIOUS: %.x-cc.model
%.x-cc.model: %.x-ccu.model  scripts/ccu2cc.py
	cat $<  |  $(PYTHON) $(word 2,$^)  >  $@

####
# obtain model-based depth-bounded model that isn't 10x larger than it should be
.PRECIOUS: %.bd.x-ccp.model
%.bd.x-ccp.model:  %.x-cc.model  scripts/pcfg2pxmodel.py
	cat $<  |  $(PYTHON) $(word 2,$^)  >  $@

.PRECIOUS: %.bogusbd.x-ccp.model
%.bogusbd.x-ccp.model:  %.x-cc.model  scripts/pcfg2pxmodel.py
	cat $<  |  sed 's/-eb/-bb/g;s/-ei/-bi/g;s/-ee/-be/g'  |  $(PYTHON) $(word 2,$^)  >  $@

#### x-cfp (hhmm) model
.PRECIOUS: %.x-cfp.model
%.x-cfp.model:  %.x-ccp.model  scripts/calc-cfp-hhmm.py  scripts/sortbyprob.pl
	cat $<  |  egrep    '^(CC|Cr)'  | $(PYTHON) $(word 2,$^)  |  perl $(word 3,$^)  >  $@
	cat $<  |  egrep -v '^(PX|CC|Cr)' >>  $@

##### x-rte (hhmm) model
.PRECIOUS: %.x-rtue.model
%.x-rtue.model:  %.x-ccp.model  bin/calc-rtue-model  scripts/sortbyprob.pl
	cat $<  |  grep -v '^PX '  | sed 's/^Cr : \(.*\^[Ll],1\) =/CC REST^R,0 : \1 REST^R,0 =/;s/ - - / -\^.,. -\^.,. /' \
		|  sed 's/CC \(.*\)\^\(.\),\(.\) : \(.*\)\^.,. \(.*\)\^.,. = \(.*\)/CC \2 \3 \1 : \4 \5 = \6/' |  $(word 2,$^)  |  perl $(word 3,$^)  >  $@
.PRECIOUS: %.x-fawp.model
%.x-fawp.model:  %.x-ccp.model  bin/calc-fawp-model  scripts/sortbyprob.pl
	cat $<  |  grep -v '^PX '  | sed 's/^Cr : \(.*\^[Ll],1\) =/CC REST^R,0 : \1 REST^R,0 =/;s/ - - / -\^.,. -\^.,. /' \
		|  sed 's/CC \(.*\)\^\(.\),\(.\) : \(.*\)\^.,. \(.*\)\^.,. = \(.*\)/CC \2 \3 \1 : \4 \5 = \6/' |  $(word 2,$^)  |  perl $(word 3,$^)  >  $@

####
%.t2m.x-rtue.model:  %.cnftrees scripts/cnftrees2rtuemodel.py  scripts/relfreq.pl  scripts/sortbyprob.pl
	cat $<  |  python3 $(word 2,$^)  |  perl $(word 3,$^)  |  perl $(word 4,$^)  >  $@

%.t2m.x-fawp.model:  %.cnftrees scripts/cnftrees2rtuemodel.py  scripts/relfreq.pl  scripts/sortbyprob.pl
	cat $<  |  python3 $(word 2,$^)  |  perl $(word 3,$^)  |  perl $(word 4,$^)  >  $@

################################################################################
#
#  5. Parsing items
#
#  to construct the following file types:
#    <x>.sents                       : sentences, one per line, consisting of only tokenized words, delimited by spaces
#    <w>.<x>_<y>_<z>_parsed.cnftrees : .cnftrees file resulting from applying parser-<z>, using parameter <y>, and model <x>.<z>.model, to <w>.sents file
#    <x>.nosm.cnftrees              : remove latent variable annotation (berkeley split-merge grammar)
#    <x>.syneval : evaluation report for parsing
#    <x>.depeval : evaluation report for syntactic dependencies
#    <x>.gapeval : evaluation report for long-distance dependencies
#
#  e.g.: make wsj22-10first.wsjTRAINberk-2sm_unked.wsjTRAINberk-2sm-mdepth_-b500,-xa_x-cfp_parsed.syneval
#				 make wsj22-393first.wsjTRAINberk-6sm_,_berk_parsed.syneval
#
################################################################################

#### obtain input sentences from linetrees
.PRECIOUS: %.sents
%.sents: %.linetrees
	cat $<  |  sed 's/(-NONE-[^)]*)//g'  \
		|  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$$//;'  \
		| sed 's/!unf! *//g' >  $@

#### obtain input sentences
.PRECIOUS: %.sents
%.sents: genmodel/$$(subst -,.,$$*).sents
	cp $< $@

.PRECIOUS: %.hysents
%.hysents: %.sents
	cat $< | perl -pe 'while ( s/(\w+)(\-)(\w+)/\1 \2 \3/g ){}; while ( s/([^ \d]+)\\(\/)([^ \d]+)/\1 \2 \3/g ){};' > $@

####
# replace rare words with UNKs from Berkeley set
%.unked.sents:  $$(basename %).sents  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*))).x-ccu.model  scripts/unkreplace.py
	cat $<  |  python3  $(word 3,$^)  $(word 2,$^)  >  $@

#### <testset>.<trainset>.<model>.(<params>_)parsed  ---->  genmodel/<testset>.sents  bin/parser-<model>  genmodel/<trainset>.<model>.model
.PRECIOUS: %parsed.output
%parsed.output: $$(basename $$(basename $$(basename %))).$$(findstring hy,$$*)sents \
		bin/parser-$$(subst .,,$$(suffix $$(basename $$*))) \
		genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))).$$(subst .,,$$(suffix $$(basename $$*))).model
	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
	@sleep 5
	cat $<  |  $(word 2,$^)  $(subst _, ,$(subst .,,$(suffix $*)))  $(word 3,$^)  >  $@

.PRECIOUS: %parsed.errlog
%parsed.errlog: $$(basename $$(basename $$(basename %))).sents \
		bin/parser-$$(subst .,,$$(suffix $$(basename $$*))) \
		genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))).$$(subst .,,$$(suffix $$(basename $$*))).model
	make $(basename $@).output 2> $@

.PRECIOUS: %enju.output
%enju.output: $$(basename $$(basename %)).sents ../enju-2.4.2/enju
	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
	@sleep 5
	$(word 2,$^) < $<  >  $@

#### obtain cnftrees by converting output using script:
%parsed.cnftrees: %parsed.output scripts/$$(lastword $$(subst -, ,$$(basename $$*)))out2cnftrees.py
	cat $<  |  python3 $(word 2,$^)  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g'  >  $@

%-cc.parsed.linetrees:  %-cc.parsed.output
	cat $<  |  sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/@//g'  >  $@

# sed magic fixes numbers... (CD 20 million) converted to (CD 20) (CD million)
# also convert things like (NP-lA-9 69 1\/4) to (NP-lA-9 (NP-lI-9 69) (NP-lA-9 1\/4))
.PRECIOUS: %.berk.parsed.cnftrees
%.berk.parsed.cnftrees:  %.berk.parsed.output scripts/killUnaries.pl
	cat $<  |  perl -pe 's/\&(?=[^\)]* )/\-/g'  |  sed 's/CD\([^A-Z ]*\) \([0-9]*\) /CD\1 \2) (CD\1 /g' |  perl -pe 's/^ *\( *//;s/ *\) *$$//;' |  perl $(word 2,$^) | perl -pe 's/\(([^ \)\(]+)(\-l[A-Z])(\-[0-9]+) ([^ \)\(]+) ([^ \)\(]+)\)/(\1\2\3 (\1-lI\3 \4) (\1-lA\3 \5))/g' >  $@
# EXAMPLE: make wsj22-10first.sm2-berk_unked.sm2-berk_,_x-cc_parsed.syneval wsj22-10first.sm2-berk_berkparsed.syneval 

.PRECIOUS: %.nosm.cnftrees
%.nosm.cnftrees: %.cnftrees scripts/removeAt.py scripts/repairToEdited.rb
	cat $<  |  perl -pe 's/\(([^ ]+)-[0-9]+ /\(\1 /g;s/\_[0-9]+//g;s/\^g//g;s/[ ]+/ /g;s/\) \)/\)\)/g'  |  python $(word 2,$^) | ruby $(word 3,$^) >  $@

# input is like: wsj23.wsj01to21-gcg-nol-1671-5sm.berk.parsed.nosm.cnftrees
.PRECIOUS: %.addl.cnftrees
%.addl.cnftrees: %.cnftrees scripts/gen-l-feats.py scripts/annotateL.py 
	$(PYTHON) $(word 2,$^) -t genmodel/wsj01to21.gcg.cnftrees > genmodel/wsj01to21.gcg.cnftrees.lfeats
	../maxent-20061005/src/opt/maxent -v -i3000 -g100 genmodel/wsj01to21.gcg.cnftrees.lfeats --model genmodel/wsj01to21.gcg.cnftrees.lmodel
	$(PYTHON) $(word 3,$^) -w genmodel/wsj01to21.gcg.cnftrees.lmodel -n $< > $@ 

#### turn cnftrees into more standard, flatter parse format
.PRECIOUS: %.evalform
%.evalform: %.cnftrees scripts/unbinarize.pl
	cat $< | sed 's/[^ \/]*\#//g' | sed 's/[^\(\) ]*://g;s/\.e[0-9]//g' | perl $(word 2,$^) | perl -p -e 's/(\([A-Z\$$\.\,\!\`\'\'']+)[-a-z\$$]+[-a-zA-Z0-9\$$]*/\1/g' > $@
#' # need this to view rest of make correctly following single-quote trickery above

#### obtain eval by running evaluator on gold and hypoth trees
%.syneval:  user-subproject-includes.txt  bin/evalb  srcmodel/new.prm  genmodel/$$(notdir $$(word 1,$$(subst ., ,$$@))).nounary.linetrees  %.cnftrees
	$(word 2,$^) -p $(word 3,$^) $(word 4,$^) $(word 5,$^) > $@

%.editeval: scripts/markRepair.rb scripts/evalEdit.rb genmodel/$$(notdir $$(word 1,$$(subst ., ,$$@))).nounary.linetrees  %.cnftrees
	cat $(word 3,$^) | ruby $(word 1,$^) > eraseme1
	cat $(word 4,$^) | ruby $(word 1,$^) > eraseme2
	ruby $(word 2,$^) eraseme1 eraseme2 > $@

%.failures:  $$(basename $$(basename %)).cnftrees  scripts/getfailures.py  genmodel/$$(subst .,,$$(suffix $$(basename $$*))).$$(subst .,,$$(suffix $$*)).model
	cat $<  |  $(PYTHON) $(word 2,$^) $(word 3,$^)  >  $@

#### obtain syntactic dependency eval on enju parser
%.enju.depeval:  scripts/depeval.py  genmodel/$$(notdir $$(word 1,$$(subst ., ,$$@))).gcg.melconts  %.enju.melconts
	$(PYTHON) $< -l $(word 2,$^) $(word 3,$^)  |  grep -n ''  >  $@

#### obtain syntactic dependency eval
%.depeval:  scripts/depeval.py  genmodel/$$(notdir $$(word 1,$$(subst ., ,$$@))).gcg.melconts  %.melconts
	$(PYTHON) $< $(word 2,$^) $(word 3,$^)  |  grep -n ''  >  $@

#### obtain syntactic dependency eval
%.hydepeval:  scripts/depeval.py  genmodel/$$(notdir $$(word 1,$$(subst ., ,$$@))).gcg.hy.melconts  %.melconts
	$(PYTHON) $< $(word 2,$^) $(word 3,$^)  |  grep -n ''  >  $@

#### obtain long-distance dependency eval
%.gapeval:  scripts/lrdepeval.py  genmodel/$$(notdir $$(word 1,$$(subst ., ,$$@))).ans.tbconts  %.tbconts
	$(PYTHON) $< $(word 2,$^) $(word 3,$^)  |  grep -n ''  >  $@
%.lrview:  scripts/lrviewer.py  %.gapeval %.output %.tbconts
	$(PYTHON) $<  $(word 2,$^) $(word 3,$^) $(word 4,$^)  |  grep -n ''  >  $@


### scores
%.scores: %.syneval
	cat $< | grep -v '^2 or' | grep -v '^    ' | grep '^[ 0-9]' | perl -na -e 'if ($$F[1]<=40) {print "$$F[0] $$F[3]\n";}' > $@

################################################################################
#
#  Misc utilities
#
################################################################################

grep.%:
	grep $(subst '.',' ',$*) src/*.cpp include/*.h ../rvtl/include/*.h -n

%.memprof: run-%
	valgrind --tool=massif --time-unit=i --max-snapshots=500 --massif-out-file=$@ -v $<

%.procprof: 
	cat user-cflags.txt > user-cflags.tmp.txt
	echo '-DNDEBUG -O3 -pg' > user-cflags.txt
	make $* -B
	gprof $* > $@
	cat user-cflags.tmp.txt > user-cflags.txt

dist-clean:
	@echo 'Do you really want to destroy all models in genmodel?  If not, CTRL-C and copy it from somewhere!'
	@sleep 5
	-rm bin/* genmodel/* */*.o ./*~ ./*~ */*.a */*.cmx */*.d ./semantic.cache pkgmodel/*
clean:
	@echo 'Do you really want to destroy all models in genmodel?  If not, CTRL-C and copy it from somewhere!'
	@sleep 5
	-rm bin/* genmodel/* */*.o ./*~ ./*~ */*.a */*.cmx */*.d ./semantic.cache
tidy:
	-rm bin/*            */*.o ./*~ ./*~ */*.a */*.cmx */*.d ./semantic.cache
