
#### default item: the main product of this directory...
genmodel/starcg-test-nounary-nolabel-40maxwords..chgcg-train-5sm..chccg-train-5sm..berk.parsed.nounary.nolabel.syneval.bootstrapsignif:

.SUFFIXES:
.SECONDEXPANSION:
SHELL = /bin/bash

# included...
include $(dir $(CURDIR))general/Makefile
include $(dir $(CURDIR))gcg/Makefile
include $(dir $(CURDIR))lvpcfg/Makefile

#### location of treebank
user-treebank-location.txt:
	echo '/home/corpora/original/chinese/chinese_treebank_6' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your treebank repository, and re-run make to continue!'
	@echo ''

#edit it to point at your treebank repository
user-chccgbank-location='/home/corpora/original/chinese/chinese_ccg/final'

%.extrops.linetrees: %.extrparplusops.linetrees
	cat $<  |  sed 's/^(\(.*\))$$/\1/g' >  $@

###Generate Chinese PTB data
PRECIOUS: genmodel/chtb%.linetrees

genmodel/chtb.trees:   user-treebank-location.txt  $$(shell cat user-treebank-location.txt)/data/utf8/bracketed  ../general/scripts/editabletrees2linetrees.pl | genmodel        
	cat $(word 2,$^)/*.fid  |  perl $(word 3,$^)  >  $@

genmodel/chtb.linetrees: genmodel/chtb.trees
	grep \(IP $^ > $@

###Generate GCG Chinese treebank
genmodel/chgcg.linetrees: genmodel/chtb.linetrees scripts/annotate-chgcg.pl
	cat $(word 1,$^) | perl $(word 2, $^) > $@ 


genmodel/annot-chgcg.linetrees: genmodel/chgcg.linetrees 
	python scripts/evaluateCoverage.py genmodel/chtb.linetrees genmodel/chgcg.linetrees genmodel/unannot-chgcg.linetrees genmodel/annot-chgcg.linetrees 27000


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

genmodel/starcg.test.nounary.nolabel.linetrees: genmodel/chccg.same.test.extrops.nounary.nolabel.linetrees  genmodel/chgcg.same.test.extrops.nounary.nolabel.linetrees
	comm -12  <(sort $(word 1,$^))  <(sort $(word 2,$^))  |  grep -v '^$$'  >  $@


#### Train a Chinese GCG parser
# divide train, test and dev by chccg's train test and dev

genmodel/chgcg.train.linetrees genmodel/chgcg.same.dev.linetrees genmodel/chgcg.same.test.linetrees genmodel/chccg.same.train.linetrees genmodel/chccg.same.dev.linetrees genmodel/chccg.same.test.linetrees: genmodel/annot-chgcg.linetrees genmodel/chccg.dev.linetrees genmodel/chccg.test.linetrees genmodel/chccg.train.linetrees
	cd scripts;\
	python divideTrainDevTest.py


