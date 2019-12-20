#### default item: the main product of this directory...
genmodel/starcg.test.nounary.nolabel.40maxwords.bootstrapsignif:
genmodel/starcg.test.nounary.nolabel.bootstrapsignif:
## WS: this should work with no special items...
genmodel/starcg-test-nounary-nolabel-40maxwords..chgcg-train-5sm..chccg-train-5sm..berk.parsed.nounary.nolabel.syneval.bootstrapsignif:

.SUFFIXES:
.SECONDEXPANSION:
SHELL := /bin/bash

# included...
include $(dir $(CURDIR))resource-general/Makefile
include $(RESOURCE-LTREES)/Makefile
#include $(dir $(CURDIR))ccgbank/Makefile
include $(RESOURCE-GCG)/Makefile
include $(RESOURCE-LVPCFG)/Makefile

#### location of treebank
user-treebank-location.txt:
	echo '/data/corpora/original/chinese/chinese_treebank_6' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your treebank repository, and re-run make to continue!'
	@echo ''

#### location of chinese treebank8
user-treebank8-location.txt:
	echo '/data/corpora/original/chinese/chinese_treebank_8' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your treebank repository, and re-run make to continue!'
	@echo ''

#edit it to point at your treebank repository
user-chccgbank-location='/data/corpora/original/chinese/chinese_ccg/final'


### using chinese treebank 8
genmodel/chtb8.trees:   user-treebank8-location.txt  $(shell cat user-treebank8-location.txt)/data/bracketed  $(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel        
	cat $(word 2,$^)/*.*  |  perl $(word 3,$^)  >  $@

genmodel/chtb8.linetrees: genmodel/chtb8.trees
	grep \(IP $^ > $@

###Generate GCG Chinese treebank
genmodel/chgcg8.linetrees: genmodel/chtb8.linetrees scripts/annotate-chgcg.pl
	cat $(word 1,$^) | perl $(word 2, $^) > $@ 


genmodel/annot-chgcg8.linetrees: genmodel/chgcg8.linetrees
	python scripts/evaluateCoverage.py genmodel/chtb8.linetrees genmodel/chgcg8.linetrees genmodel/unannot-chgcg8.linetrees $@ 68000 genmodel/annot-chgcg8-chtb.linetrees genmodel/chtb8.linetrees

genmodel/annot-chgcg8-chtb.linetrees: genmodel/annot-chgcg8.linetrees
	noop



%.extrops.linetrees: %.extrparplusops.linetrees
	cat $<  |  sed 's/^(\(.*\))$$/\1/g' >  $@

###Generate Chinese PTB data
PRECIOUS: genmodel/chtb%.linetrees

genmodel/chtb.trees:   user-treebank-location.txt  $$(shell cat user-treebank-location.txt)/data/utf8/bracketed  $(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel        
	cat $(word 2,$^)/*.fid  |  perl $(word 3,$^)  >  $@

genmodel/chtb.linetrees: genmodel/chtb.trees
	grep \(IP $^ > $@

###Generate GCG Chinese treebank
genmodel/chgcg.linetrees: genmodel/chtb.linetrees scripts/annotate-chgcg.pl
	cat $(word 1,$^) | perl $(word 2, $^) > $@ 


genmodel/annot-chgcg.linetrees: genmodel/chgcg.linetrees
	python scripts/evaluateCoverage.py genmodel/chtb.linetrees genmodel/chgcg.linetrees genmodel/unannot-chgcg.linetrees $@ 27000 genmodel/annot-chgcg-chtb.linetrees genmodel/chtb.linetrees



###Chinese CCG
##divide CCG dev, train, and test section according to Tse and Curran (2012)
genmodel/chccg.dev.linetrees:
	mkdir genmodel/chccg_dev;\
	cp -r $(user-chccgbank-location)/09/* genmodel/chccg_dev;\
	cp -r $(user-chccgbank-location)/11/chtb_1148.fid $(user-chccgbank-location)/11/chtb_1149.fid $(user-chccgbank-location)/11/chtb_1150.fid $(user-chccgbank-location)/11/chtb_1151.fid genmodel/chccg_dev;\
	cp $(user-chccgbank-location)/29/chtb_298* $(user-chccgbank-location)/29/chtb_299* genmodel/chccg_dev;\
	rm -rf genmodel/chccg_dev/chtb_2980.fid;\
	cp $(user-chccgbank-location)/30/chtb_300* $(user-chccgbank-location)/30/chtb_301* $(user-chccgbank-location)/30/chtb_302* genmodel/chccg_dev;\
	cat genmodel/chccg_dev/* > genmodel/chccg.dev.fid;\
	cat genmodel/chccg.dev.fid | perl ../ccgbank/scripts/ccglinetrees2nicelinetrees.pl  >  $@

genmodel/chccg.test.linetrees:
	mkdir genmodel/chccg_test;\
	cp -r $(user-chccgbank-location)/08/chtb_082* $(user-chccgbank-location)/08/chtb_083* $(user-chccgbank-location)/08/chtb_084* $(user-chccgbank-location)/08/chtb_085* $(user-chccgbank-location)/08/chtb_086* $(user-chccgbank-location)/08/chtb_087* genmodel/chccg_test;\
	cp $(user-chccgbank-location)/08/chtb_0816.fid $(user-chccgbank-location)/08/chtb_0817.fid $(user-chccgbank-location)/08/chtb_0818.fid $(user-chccgbank-location)/08/chtb_0819.fid genmodel/chccg_test;\
	cp $(user-chccgbank-location)/11/chtb_1137* $(user-chccgbank-location)/11/chtb_1138* $(user-chccgbank-location)/11/chtb_1139* $(user-chccgbank-location)/11/chtb_1140* $(user-chccgbank-location)/11/chtb_1141* $(user-chccgbank-location)/11/chtb_1142* $(user-chccgbank-location)/11/chtb_1143* $(user-chccgbank-location)/11/chtb_1144* $(user-chccgbank-location)/11/chtb_1145* $(user-chccgbank-location)/11/chtb_1146* $(user-chccgbank-location)/11/chtb_1147* genmodel/chccg_test;\
	cp $(user-chccgbank-location)/30/chtb_303* $(user-chccgbank-location)/30/chtb_304* $(user-chccgbank-location)/30/chtb_305* $(user-chccgbank-location)/30/chtb_306* $(user-chccgbank-location)/30/chtb_307* $(user-chccgbank-location)/30/chtb_308* $(user-chccgbank-location)/30/chtb_309* genmodel/chccg_test;\
	cp -r $(user-chccgbank-location)/31/* genmodel/chccg_test;\
	cat genmodel/chccg_test/* > genmodel/chccg.test.fid;\
	cat genmodel/chccg.test.fid | perl ../ccgbank/scripts/ccglinetrees2nicelinetrees.pl  >  $@

genmodel/chccg.train.linetrees:
	mkdir genmodel/chccg_train;\
	cp -r $(user-chccgbank-location)/01/* $(user-chccgbank-location)/02/* $(user-chccgbank-location)/03/* $(user-chccgbank-location)/04/* $(user-chccgbank-location)/05/* $(user-chccgbank-location)/06/* $(user-chccgbank-location)/07/* $(user-chccgbank-location)/20/* $(user-chccgbank-location)/21/* $(user-chccgbank-location)/22/* $(user-chccgbank-location)/23/* $(user-chccgbank-location)/24/* $(user-chccgbank-location)/25/* $(user-chccgbank-location)/26/*  $(user-chccgbank-location)/27/* $(user-chccgbank-location)/28/* $(user-chccgbank-location)/29/* $(user-chccgbank-location)/10/* $(user-chccgbank-location)/08/chtb_080* $(user-chccgbank-location)/08/chtb_081* genmodel/chccg_train;\
	rm -rf genmodel/chccg_train/chtb_0816* genmodel/chccg_train/chtb_0817* genmodel/chccg_train/chtb_0818* genmodel/chccg_train/chtb_0819*;\
	cp -r $(user-chccgbank-location)/11/chtb_110* $(user-chccgbank-location)/11/chtb_111* $(user-chccgbank-location)/11/chtb_112* $(user-chccgbank-location)/11/chtb_113* genmodel/chccg_train;\
	rm -rf genmodel/chccg_train/chtb_1137* genmodel/chccg_train/chtb_1138* genmodel/chccg_train/chtb_1139*;\
	rm -rf genmodel/chccg_train/chtb_298* genmodel/chccg_train/chtb_299*;\
	cp $(user-chccgbank-location)/29/chtb_2980.fid genmodel/chccg_train/;\
	cat genmodel/chccg_train/* > genmodel/chccg.train.fid;\
	cat genmodel/chccg.train.fid | perl ../ccgbank/scripts/ccglinetrees2nicelinetrees.pl  >  $@

genmodel/chccg.test1.linetrees:
	cat genmodel/chccg.test1.fid | perl ../ccgbank/scripts/ccglinetrees2nicelinetrees.pl  >  $@

#### Train a Chinese GCG parser
# divide train, test and dev by chccg's train test and dev

genmodel/chgcg.same.train.linetrees genmodel/chgcg.same.dev.linetrees genmodel/chgcg.same.test.linetrees genmodel/chccg.same.train.linetrees genmodel/chccg.same.dev.linetrees genmodel/chccg.same.test.linetrees: genmodel/annot-chgcg.linetrees genmodel/chccg.dev.linetrees genmodel/chccg.test.linetrees genmodel/chccg.train.linetrees
	python scripts/divideTrainDevTest.py $(word 1, $^)


#train a grammar with the berkeley parser
%.5sm.gr: %.extrparplusops.linetrees
	java -cp berkeley/berkeleyParser.jar edu.berkeley.nlp.PCFGLA.GrammarTrainer -SMcycles 5 -path $< -out $@ -treebank SINGLEFILE
%.5sm.gcg.parse: %.chgcg-train-5sm.berk.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@
%.5sm.gcg.parse: %5sm.gcg.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@

%.5sm.ccg.parse: %.chccg-train-5sm.berk.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@
%.5sm.ccg.same.parse: %.chccg-same-5sm.berk.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@
%.parse: %.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@

###### use for LREC 2018
# same train use the training set as chccg
%.5sm.same.gcg.parse: %.chgcg-same-train-5sm.berk.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@

# big train uses chtb8 - gapeval test sets as training set
%.5sm.big.gcg.parse: %.chgcg-big-train-5sm.berk.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@



genmodel/chgcg.same.test.5sm.parse.linetrees: genmodel/chgcg.same.test.5sm.gcg.parse
	python scripts/readParse.py $< $@  

%.parse.linetrees: %.ccg.parse 
	python scripts/readParse.py $< $@  

%.same-parse.linetrees: %.ccg.same.parse 
	python scripts/readParse.py $< $@  

%.parse.linetrees: %.parse 
	python scripts/readParse.py $< $@  

#compare ccg and gcg

genmodel/starcg.test.nounary.nolabel.linetrees: genmodel/chccg.same.test.extrops.nounary.nolabel.linetrees  genmodel/chgcg.same.test.extrops.nounary.nolabel.linetrees
	comm -12  <(sort $(word 1,$^))  <(sort $(word 2,$^))  |  grep -v '^$$'  >  $@

#make genmodel/starcg.test.nounary.nolabel.5sm.chgcg.parse.nounary.nolabel.linetree
genmodel/starcg.test.5sm.chgcg.parse.nounary.nolabel.linetrees: genmodel/starcg.test.nounary.nolabel.linetrees genmodel/chgcg.same.test.5sm.parse.nounary.nolabel.linetrees
	python scripts/generateSharedParseFile.py $(word 1, $^) $(word 2, $^) $@ 

genmodel/starcg.test.5sm.chccg.parse.nounary.nolabel.linetrees: genmodel/starcg.test.nounary.nolabel.linetrees genmodel/chccg.same.test.5sm.parse.nounary.nolabel.linetrees
	python scripts/generateSharedParseFile.py $(word 1, $^) $(word 2, $^) $@ 

genmodel/starcg.test.5sm.chccg.same-parse.nounary.nolabel.linetrees: genmodel/starcg.test.nounary.nolabel.linetrees genmodel/chccg.same.test.5sm.same-parse.nounary.nolabel.linetrees
	python scripts/generateSharedParseFile.py $(word 1, $^) $(word 2, $^) $@ 


#make genmodel/starcg.test.chgcg.nounary.nolabel.syneval
%.chgcg.parse.nounary.nolabel.40maxwords.syneval: bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.40maxwords.linetrees %.5sm.chgcg.parse.nounary.nolabel.40maxwords.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

%.chccg.parse.nounary.nolabel.40maxwords.syneval: bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.40maxwords.linetrees %.5sm.chccg.parse.nounary.nolabel.40maxwords.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

%.chccg.same-parse.nounary.nolabel.40maxwords.syneval: bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.40maxwords.linetrees %.5sm.chccg.same-parse.nounary.nolabel.40maxwords.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

%.chgcg.parse.nounary.nolabel.syneval: bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.linetrees %.5sm.chgcg.parse.nounary.nolabel.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

%.chccg.parse.nounary.nolabel.syneval: bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.linetrees %.5sm.chccg.parse.nounary.nolabel.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

%.chccg.same-parse.nounary.nolabel.syneval: bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.linetrees %.5sm.chccg.same-parse.nounary.nolabel.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@


#syneval of test section
genmodel/chgcg.test.parse.40maxwords.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm genmodel/chgcg.test.extrops.40maxwords.linetrees genmodel/chgcg.same.test.5sm.parse.40maxwords.linetrees  
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

genmodel/chccg.test.parse.40maxwords.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm genmodel/chccg.test.extrops.40maxwords.linetrees genmodel/chccg.same.test.5sm.parse.40maxwords.linetrees  
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

genmodel/chccg.test.same-parse.40maxwords.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm genmodel/chccg.test.extrops.40maxwords.linetrees genmodel/chccg.same.test.5sm.same-parse.40maxwords.linetrees  
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@


%.chgcg.nounary.nolabel.nopunc.40maxwords.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.nopunc.40maxwords.linetrees %.5sm.chgcg.parse.nounary.nolabel.nopunc.40maxwords.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

%.chccg.nounary.nolabel.nopunc.40maxwords.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.nopunc.40maxwords.linetrees %.5sm.chccg.parse.nounary.nolabel.nopunc.40maxwords.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@


genmodel/chgcg.test.parse.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm genmodel/chgcg.test.extrops.linetrees genmodel/chgcg.same.test.5sm.parse.linetrees  
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

genmodel/chccg.test.parse.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm genmodel/chccg.test.extrops.linetrees genmodel/chccg.same.test.5sm.parse.linetrees  
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

genmodel/chccg.test.same-parse.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm genmodel/chccg.test.extrops.linetrees genmodel/chccg.same.test.5sm.same-parse.linetrees  
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@


%.chgcg.nounary.nolabel.nopunc.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.nopunc.linetrees %.5sm.chgcg.parse.nounary.nolabel.nopunc.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

%.chccg.nounary.nolabel.nopunc.syneval: $(RESOURCE-GENERAL)/bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm %.nounary.nolabel.nopunc.linetrees %.5sm.chccg.parse.nounary.nolabel.nopunc.linetrees 
	$(word 1,$^) -p $(word 2,$^) $(word 3, $^) $(word 4, $^) > $@

###################
#make genmodel/starcg.test.nounary.nolabel.bootstrapsignif
%.nounary.nolabel.40maxwords.bootstrapsignif: %.chccg.parse.nounary.nolabel.40maxwords.syneval %.chgcg.parse.nounary.nolabel.40maxwords.syneval
	perl $(LTREES-SCRIPTS)/compare.pl $(word 1, $^) $(word 2, $^) > $@ 

%.same-train.nounary.nolabel.40maxwords.bootstrapsignif: %.chccg.same-parse.nounary.nolabel.40maxwords.syneval %.chgcg.parse.nounary.nolabel.40maxwords.syneval
	perl $(LTREES-SCRIPTS)/compare.pl $(word 1, $^) $(word 2, $^) > $@ 

%.nounary.nolabel.bootstrapsignif: %.chccg.parse.nounary.nolabel.syneval %.chgcg.parse.nounary.nolabel.syneval
	perl $(LTREES-SCRIPTS)/compare.pl $(word 1, $^) $(word 2, $^) > $@ 

%.same-train.nounary.nolabel.bootstrapsignif: %.chccg.same-parse.nounary.nolabel.syneval %.chgcg.parse.nounary.nolabel.syneval
	perl $(LTREES-SCRIPTS)/compare.pl $(word 1, $^) $(word 2, $^) > $@ 




##gap eval in chinese
## Generate chtb.test.linetrees according to gcg test set

genmodel/chtb.train.linetrees genmodel/chtb.dev.linetrees genmodel/chtb.test.linetrees: genmodel/chtb.linetrees
	python scripts/divideTrainDevTestChtb.py $(word 1, $^) genmodel/chtb.train.linetrees genmodel/chtb.dev.linetrees genmodel/chtb.test.linetrees

#short passive voice 
genmodel/chtb.sb.linetrees:genmodel/chtb.test.linetrees
	grep "SB " $^ > $@

#long passive voice
genmodel/chtb.lb.linetrees:genmodel/chtb.test.linetrees
	grep "LB " $^ > $@

genmodel/extra.chgcg.linetrees: 
	python3 scripts/selectComplement.py genmodel/annot-chgcg.linetrees genmodel/annot-chgcg8.linetrees $@
genmodel/extra.chtb.linetrees:
	python3 scripts/selectComplement.py genmodel/annot-chgcg-chtb.linetrees  genmodel/annot-chgcg8-chtb.linetrees $@

genmodel/chtb.big.test.linetrees:genmodel/chtb.test.linetrees genmodel/extra.chtb.linetrees
	cat $(word 1, $^) $(word 2, $^) > $@

genmodel/chgcg.big.test.linetrees:genmodel/chgcg.same.test.linetrees genmodel/extra.chgcg.linetrees
	cat $(word 1, $^) $(word 2, $^) > $@

# gap constructions to examine
#genmodel/gapeval.sbjrc.chgcg.linetrees:
#	grep "(V-gN (V-aN " genmodel/chgcg.same.test.linetrees |sort -r |head -130 > $@

genmodel/gapeval.extractFromEmb.chgcg.linetrees:
	grep "(V-aN-gN (V-aN-bV" genmodel/chgcg.big.test.linetrees > $@

#genmodel/gapeval.extractFromObj.chgcg.linetrees:
#	grep "(V-aN-gN (V-aN-b{V-aN}-gN" genmodel/chgcg.big.test.linetrees > $@

genmodel/gapeval.dobjrc.chgcg.linetrees:
	grep "(V-aN-gN (V-aN-bN" genmodel/chgcg.big.test.linetrees > $@

genmodel/gapeval.iobjrc.chgcg.linetrees:
	grep "(V-aN-gN (V-aN-b{V-aN}" genmodel/chgcg.big.test.linetrees > $@

genmodel/gapeval.objrc.chgcg.big.linetrees:genmodel/gapeval.dobjrc.chgcg.linetrees genmodel/gapeval.iobjrc.chgcg.linetrees
	cat $(word 1, $^) $(word 2, $^) > $@


genmodel/gapeval.objrc.chgcg.linetrees: genmodel/gapeval.objrc.chgcg.big.linetrees
	cat $^ |awk '!seen[$$0]++'| sort -R |head -n 200 > $@



genmodel/gapeval.passive_do.chgcg.linetrees:
	grep "(E-aN-bN (V-aN" genmodel/chgcg.big.test.linetrees |sort -R |head -300 > $@

genmodel/gapeval.passive_io.chgcg.linetrees:
	grep "E-aN-bN-bN" genmodel/chgcg.big.test.linetrees > $@
	grep "E-aN-b{V-aN}-bN" genmodel/chgcg.big.test.linetrees >> $@

genmodel/gapeval.tpcrc.chtb.linetrees:
	grep "NP-TPC (-NONE-" genmodel/chtb.big.test.linetrees |sort -R |head -140 > $@

genmodel/gapeval.topic.chtb.linetrees:
	grep "NP-TPC-" genmodel/chtb.big.test.linetrees |sort -R |head -140 > $@

genmodel/gapeval.foc.chtb.linetrees:
	grep "NP-FOC-" genmodel/chtb.big.test.linetrees |sort -R |head -105 > $@

#genmodel/gapeval.rnr.chtb.linetrees:
#	grep "RNR" genmodel/chtb.big.test.linetrees |sort -R |head -130 > $@

%.chgcg.linetrees:%.chtb.linetrees genmodel/chgcg.big.test.linetrees
	python3 scripts/selectMatchTrees.py $(word 1, $^) $(word 2, $^) $@

%.chtb.linetrees:%.chgcg.linetrees genmodel/chtb.big.test.linetrees
	python3 scripts/selectMatchTrees.py $(word 1, $^) $(word 2, $^) $@

genmodel/gapeval.sbjrc.chgcg.anw:
	cat genmodel/gapeval.sbjrc.chgcg.final.linetrees |python3 scripts/genDepsSemEval16_fix_sbjgap.py > $@

genmodel/gapeval.objrc.chgcg.anw:
	cat genmodel/gapeval.objrc.chgcg.final.linetrees |python3 scripts/genDepsSemEval16_fix_sbjgap.py > $@

genmodel/gapeval.tpcrc.chgcg.anw:
	cat genmodel/gapeval.tpcrc.chgcg.final.linetrees |python3 scripts/genDepsSemEval16_fix_sbjgap.py > $@

genmodel/gapeval.topic.chgcg.anw:
	cat genmodel/gapeval.topic.chgcg.final.linetrees |python3 scripts/genDepsSemEval16_fix_topic.py > $@

%.gcg.deps: %.gcg.parse
	cat $^ |python3 scripts/genDepsSemEval16_fix_topic.py > $@


# convert penn trees to stanford dependencies
%.sd.deps: %.chtb.linetrees
	java -cp stanford_parser/stanford-parser-full-2017-06-09/stanford-parser.jar  edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -basic -treeFile $^ > $@

%.final.sd.deps: %.chtb.final.linetrees
	java -cp stanford_parser/stanford-parser-full-2017-06-09/stanford-parser.jar  edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -basic -treeFile $^ > $@


# convert stanford dependencies for gap eval
%.sd2gcg.deps: %.sd.deps
	python3 scripts/SDdeps2GcgDeps.py $^ > $@


# extract predicate-argument deps from gcg trees
%.gcg.deps: %.chgcg.linetrees
	cat $^ |python3 scripts/genGcgDeps.py > $@

%.gcg.deps: %.gcg.parse
	cat $^ |python3 scripts/genGcgDeps.py > $@

genmodel/gapeval.all.test.chgcg.linetrees:
	cat gapeval/gapeval*final.linetrees > $@

genmodel/chgcg.big.train.linetrees:genmodel/gapeval.all.test.chgcg.linetrees genmodel/annot-chgcg8.linetrees
	python3 scripts/selectComplement.py $(word 1, $^) $(word 2, $^) $@

%.chgcg.5sm.big.gcg.results: %.anw %.chgcg.5sm.big.gcg.deps
	python3 scripts/gapevalEvaluation.py $(word 1, $^) $(word 2, $^) > $@

%.final.gapeval.results: %.anw %.final.sd2gcg.deps
	python3 scripts/gapevalEvaluation.py $(word 1, $^) $(word 2, $^) > $@

%.5sm.big.gcg.gapeval.results: %.anw %.chgcg.final.5sm.big.gcg.deps
	python3 scripts/gapevalEvaluation.py $(word 1, $^) $(word 2, $^) > $@


%.chgcg.5sm.same.gcg.results: %.anw %.chgcg.5sm.same.gcg.deps
	python3 scripts/gapevalEvaluation.py $(word 1, $^) $(word 2, $^) > $@

genmodel/big.train.chgcg.linetrees genmodel/big.train.chtb.linetrees:genmodel/chgcg.big.train.linetrees genmodel/chtb8.linetrees
	python3 scripts/selectMatchTrees.py $(word 1, $^) $(word 2, $^) genmodel/big.train.chgcg.linetrees genmodel/big.train.chtb.linetrees > unmatched.big.train.linetrees

%.chtb.final.linetrees :%.chgcg.final.linetrees genmodel/chtb8.linetrees
	python3 scripts/exactMatchTrees.py $(word 1, $^) $(word 2, $^) $@ 
