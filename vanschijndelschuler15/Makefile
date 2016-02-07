################################################################################
##                                                                            ##
##  This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                            ##
##  ModelBlocks is free software: you can redistribute it and/or modify       ##
##  it under the terms of the GNU General Public License as published by      ##
##  the Free Software Foundation, either version 3 of the License, or         ##
##  (at your option) any later version.                                       ##
##                                                                            ##
##  ModelBlocks is distributed in the hope that it will be useful,            ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of            ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             ##
##  GNU General Public License for more details.                              ##
##                                                                            ##
##  You should have received a copy of the GNU General Public License         ##
##  along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.      ##
##                                                                            ##
################################################################################

################################################################################
#
#  i. Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

naacl15: prereqs \
	scripts/intersectComplexity.r \
	scripts/querydataframe.py \
	dundee.wsj02to21-nodashtags-5sm-bd.x-efabp.-c_-b5000.fp.fulltoks \
	dundee.wsj02to21-gcg14-3sm-bd.x-efabp.-c_-b5000.fp.fulltoks \
	dundee.wsj02to21-nodashtags-5sm-bd.x-efabp.-c_-b5000.gp.fulltoks \
	dundee.wsj02to21-gcg14-3sm-bd.x-efabp.-c_-b5000.gp.fulltoks
	$(word 2,$^) $(word 4,$^) $(word 5,$^)
	$(word 2,$^) $(word 6,$^) $(word 7,$^)
	python $(word 3,$^) --input $(word 4,$^) --i totsurp | sed '1s/.*/ptbsurp/' | paste -d' ' $(word 5,$^) - > $(word 5,$^).tmp
	mv $(word 5,$^){.tmp,}
	python $(word 3,$^) --input $(word 4,$^) --i cumtotsurp | sed '1s/.*/cumptbsurp/' | paste -d' ' $(word 5,$^) - > $(word 5,$^).tmp
	mv $(word 5,$^){.tmp,}
	python $(word 3,$^) --input $(word 5,$^) --i totsurp | sed '1s/.*/gcgsurp/' | paste -d' ' $(word 5,$^) - > $(word 5,$^).tmp
	mv $(word 5,$^){.tmp,}
	python $(word 3,$^) --input $(word 5,$^) --i cumtotsurp | sed '1s/.*/cumgcgsurp/' | paste -d' ' $(word 5,$^) - > $(word 5,$^).tmp
	mv $(word 5,$^){.tmp,}
	python $(word 3,$^) --input $(word 6,$^) --i totsurp | sed '1s/.*/ptbsurp/' | paste -d' ' $(word 7,$^) - > $(word 7,$^).tmp
	mv $(word 7,$^){.tmp,}
	python $(word 3,$^) --input $(word 6,$^) --i cumtotsurp | sed '1s/.*/cumptbsurp/' | paste -d' ' $(word 7,$^) - > $(word 7,$^).tmp
	mv $(word 7,$^){.tmp,}
	python $(word 3,$^) --input $(word 7,$^) --i totsurp | sed '1s/.*/gcgsurp/' | paste -d' ' $(word 7,$^) - > $(word 7,$^).tmp
	mv $(word 7,$^){.tmp,}
	python $(word 3,$^) --input $(word 7,$^) --i cumtotsurp | sed '1s/.*/cumgcgsurp/' | paste -d' ' $(word 7,$^) - > $(word 7,$^).tmp
	mv $(word 7,$^){.tmp,}
	make $(basename $(word 5,$^)).-fl_-fsc_-fse_-ff_-bobyqa.arg.dundeeeval
	make $(basename $(word 5,$^)).-fl_-fsc_-fse_-ff_-nlminb.arg.dundeeeval
	make $(basename $(word 7,$^)).-fl_-fsc_-fse_-ff_-bobyqa.arg.dundeeeval
	make $(basename $(word 7,$^)).-fl_-fsc_-fse_-ff_-nlminb.arg.dundeeeval

# included...
include $(dir $(CURDIR))general/Makefile
include $(dir $(CURDIR))gcg/Makefile
include $(dir $(CURDIR))lvpcfg/Makefile
include $(dir $(CURDIR))lcparse/Makefile

prereqs: user-bnc-directory.txt user-treebank-directory.txt user-dundee-directory.txt user-tokenizer-directory.txt user-kenlm-directory.txt user-kenlm-model-directory.txt

DUNDEESECTS = 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
DUNDEESUBJS = sa sb sc sd se sf sg sh si sj
BROWNTRAINSECTS = cf cg ck cl cm cn cp cr
BNCTRAINDIRS = A B C D E F G H J
GIGASECTS = afp_eng apw_eng cna_eng ltw_eng nyt_eng xin_eng

#### R packages
#rpkgs:
#	#Frustratingly, this doesn't work because of bugs in R
#	# just install the required packages (found in scripts/installR.r) manually in the *default* location
#	# DO NOT use R_LIBS in .R_ENVIRON since that's what triggers the bug!
#	# More info: https://github.com/hadley/devtools/issues/166
#	# More info (last post): http://r.789695.n4.nabble.com/install-packages-fails-if-libPaths-set-td4647797.html
#	mkdir rpkgs
#	scripts/installR.r
#	echo 'R_LIBS="rpkgs"' > .Renviron

################################################################################
#
#  1. N-Gram stuff
#
################################################################################

#### directory of bnc
user-bnc-directory.txt:
	echo '/home/corpora/original/english/bnc' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your bnc directory, and re-run make to continue!'
	@echo ''

#### directory of dundee
user-dundee-directory.txt:
	echo '/home/corpora/original/english/dundee' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your dundee directory, and re-run make to continue!'
	@echo ''

#### directory of gigaword-4
user-gigaword4-directory.txt:
	echo '/home/corpora/original/english/gigaword-4' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your gigaword-4 directory, and re-run make to continue!'
	@echo ''

#### directory of kenlm
user-kenlm-directory.txt:
	echo '/home/compling/kenlm_git' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your kenlm directory, and re-run make to continue!'
	@echo ''

#### directory of kenlm models
user-kenlm-model-directory.txt:
	echo '/home/compling/kenlm_models' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your kenlm models directory, and re-run make to continue!'
	@echo ''

#### directory of srilm
user-srilm-directory.txt:
	echo '/home/compling/srilm/bin' > $@
	echo 'i686-m64' >> $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your srilm binaries directory, and re-run make to continue!'
	@echo ''

#### directory of tokenizer
user-tokenizer-directory.txt:
	echo '/home/compling/extended_penn_tokenizer' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your tokenizer directory, and re-run make to continue!'
	@echo 'Tokenizer is available at:'
	@echo 'https://github.com/vansky/extended_penn_tokenizer'
	@echo ''

SRILM = $(shell head -1 user-srilm-directory.txt)
SRILMSUB = $(shell tail -1 user-srilm-directory.txt)

define BNCDIRMACRO
.PRECIOUS: genmodel/bnc$(1).sents
genmodel/bnc$(1).sents: user-bnc-directory.txt $$(shell cat user-bnc-directory.txt)/Texts/$(1)/*/* | genmodel
	cat $$(shell cat user-bnc-directory.txt)/Texts/$(1)/*/* | sed 's|<teiHeader.*</teiHeader>||g;s|<[^<>]*>||g;s|^\s*\n||g;/^$$$$/d;s|  *| |g;' > $$@
endef

$(foreach dir,$(BNCTRAINDIRS),$(eval $(call BNCDIRMACRO,$(dir))))

.PRECIOUS: genmodel/bncTRAIN.sents
genmodel/bncTRAIN.sents: $(foreach sect,$(BNCTRAINDIRS),genmodel/bnc$(sect).sents) | genmodel
	cat $^ > $@

## for brown corpus
.PRECIOUS: genmodel/brown%.linetrees
genmodel/brown%.linetrees: user-treebank-directory.txt $$(shell cat user-treebank-directory.txt)/parsed/mrg/brown/% scripts/editabletrees2linetrees.pl | genmodel
	cat $(word 2,$^)/*.mrg | perl $(word 3,$^) > $@

genmodel/brownTRAIN.linetrees: $(foreach sect,$(BROWNTRAINSECTS),genmodel/brown$(sect).linetrees) | genmodel
	cat $^ > $@


genmodel/broadcoveragetraining.sents: user-tokenizer-directory.txt $(cat user-tokenizer-directory.txt)/ptb_tokenizer.sed \
					genmodel/brownTRAIN.sents genmodel/wsj02to21.sents genmodel/bncTRAIN.sents genmodel/dundee.sents | genmodel
	cat $(word 5,$^) | sed "s/&bquo;/\'/g;s/&equo;/\'/g;s/&hellip;/.../g;s/&percnt;/%/g;s/&ndash;/-/g;s/&amp;/\&/g;s/&mdash;/--/g;" | \
	$(word 2,$^) > genmodel/tmp.bnc
	$(word 2,$^) $(word 6,$^) > genmodel/tmp.dundee
	cat $(word 3,$^) $(word 4,$^) genmodel/tmp.bnc genmodel/tmp.dundee | grep -v '\*x\*' | sed 's/  */ /g;' > $@
	rm -f genmodel/tmp.bnc genmodel/tmp.dundee

# Doesn't seem to work well with gigaword; possibly too big?
#genmodel/%.revsents: genmodel/%.sents | genmodel
#	cat $^ | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/ <LINEBREAK> /g' > $@.tmp #the idiocy of BSD sed (e.g. on OSX) requires the dumb formulation of this sed command
#	tac -s ' ' $@.tmp | sed 's/ <LINEBREAK> /\n/g;s/  */ /g;' > $@
#	rm -f $@.tmp

.PRECIOUS: genmodel/dundee%.sents
genmodel/dundee%.sents: user-dundee-directory.txt $$(shell cat user-dundee-directory.txt)/tx%wrdp.dat scripts/builddundeecorpus.py scripts/convert_latin-1.py | genmodel
	cat $(word 2,$^) | python $(word 3,$^) > $@.raw
	cat $@.raw | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" | python3 $(word 4,$^) > $@
	rm -f $@.raw

.PRECIOUS: genmodel/dundee%.notoksents
#non-tokenized version of dundee (for ngram calculation)
genmodel/dundee%.notoksents: user-dundee-directory.txt $$(shell cat user-dundee-directory.txt)/tx%wrdp.dat scripts/buildnotokdundeecorpus.py scripts/convert_latin-1.py | genmodel
	cut -d" " -f1 < $(word 2,$^) | python $(word 3,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" | python3 $(word 4,$^) > $@

.PRECIOUS: genmodel/dundee.sents
genmodel/dundee.sents: $(foreach sect,$(DUNDEESECTS),genmodel/dundee$(sect).sents) | genmodel
	cat $^ > $@

.PRECIOUS: genmodel/dundee.notoksents
genmodel/dundee.notoksents: $(foreach sect,$(DUNDEESECTS),genmodel/dundee$(sect).notoksents) | genmodel
	cat $^ > $@

.PRECIOUS: genmodel/gigaword4%.sents
genmodel/gigaword4%.sents: user-gigaword4-directory.txt $$(shell cat user-gigaword4-directory.txt)/data/$$*/*.gz scripts/preprocess_gigaword.py | genmodel
	mkdir tmp
	cp $(shell cat user-gigaword4-directory.txt)/data/$*/*.gz tmp/
	gunzip tmp/*
	cat tmp/* > $@.tmp
	python scripts/preprocess_gigaword.py $@.tmp > $@
	rm -rf tmp $@.tmp

.PRECIOUS: genmodel/gigaword4.sents
genmodel/gigaword4.sents: $(foreach sect,$(GIGASECTS),genmodel/gigaword4$(sect).sents) | genmodel
	cat $^ | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^


.PRECIOUS: %.sents
%.sents: genmodel/$$*.sents | genmodel
	cp $< $@

.PRECIOUS: %.bz2
%.bz2: %
	#careful, this removes the source after compressing a copy
	#saves space, but it makes the order of the dependencies in a make item crucial; e.g. revsents.bz2 must come before sents.bz2
	bzip2 $<

################################################################################
#
#  2. Ngram stuff
#     For distributional/reproduction purposes, point user-kenlm-model-directory.txt to genmodel
#     CoCoMo Lab members should just point to the premade kenlm model in /home/compling/kenlm_models
#
################################################################################

.PRECIOUS: genmodel/%.fw.kenlm
# genmodel/gigaword4.5.fw.kenlm
# especially good for unigram models since kenlm doesn't permit those to be made into binaries
genmodel/%.fw.kenlm: user-kenlm-directory.txt $$(shell cat user-kenlm-directory.txt)/bin/lmplz \
			genmodel/$$(basename $$*).sents.bz2
	mkdir kenlmtmp
	$(word 2,$^) -o $(subst .,,$(suffix $*)) -T kenlmtmp -S 50% < $(word 3,$^) > $@
	rm -rf kenlmtmp

.PRECIOUS: genmodel/%.kenlm.bin
# genmodel/gigaword4.5.fw.kenlm.bin
# uses array pointer compression
genmodel/%.kenlm.bin: user-kenlm-directory.txt $$(shell cat user-kenlm-directory.txt)/bin/build_binary \
			genmodel/$$*.kenlm
	mkdir kenlmtmp
	$(word 2,$^) -a 64 -T kenlmtmp -S 1G trie $(word 3,$^) $@
	rm -rf kenlmtmp

.PRECIOUS: genmodel/%.kenlm.tinybin
# genmodel/gigaword4.5.fw.kenlm.tinybin
# smaller memory footprint, but slower
# uses array pointer compression and quantization
genmodel/%.kenlm.tinybin: user-kenlm-directory.txt $$(shell cat user-kenlm-directory.txt)/bin/build_binary \
			genmodel/$$*.kenlm
	mkdir kenlmtmp
	$(word 2,$^) -a 64 -q 8 -b 8 -T kenlmtmp -S 1G trie $(word 3,$^) $@
	rm -rf kenlmtmp

################################################################################

.PRECIOUS: genmodel/dundee.ngramprobtoks
# genmodel/dundee.ngramprobtoks
genmodel/dundee.ngramprobtoks: user-kenlm-directory.txt  user-kenlm-model-directory.txt \
				$$(shell cat user-kenlm-directory.txt)/bin/query \
				genmodel/dundee.notoksents \
				scripts/calcngramprobtoks.kenlm.py  $$(shell cat user-kenlm-model-directory.txt)/gigaword4.5.fw.kenlm.bin | genmodel
	$(word 3,$^) -\n $(word 6,$^) < $(word 4,$^) | python3 $(word 5,$^) --fwprobs - > $@

################################################################################
#
#  3. Eye-tracking data formatting items
#
################################################################################

#### obtain lexicon of terms that appear more than 5 times
.PRECIOUS: %.wordcounts
%.wordcounts: %.sents scripts/buildLexicon.py
	python3 $(word 2,$^) $< -u5 > $@

## for dundee eyetracking corpus
.PRECIOUS: genmodel/dundee.%.textdata
genmodel/dundee.%.textdata: user-dundee-directory.txt $$(shell cat user-dundee-directory.txt)/tx%wrdp.dat | genmodel
	echo 'WORD ' > $@
	cat $(word 2,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" >> $@

genmodel/dundee.textdata:  $(foreach sect,$(DUNDEESECTS),genmodel/dundee.$(sect).textdata) | genmodel
	cat $^  >  $@

.PRECIOUS: genmodel/dundee.%.eventdata
genmodel/dundee.%.eventdata: user-dundee-directory.txt $$(shell cat user-dundee-directory.txt)/%ma1p.dat | genmodel
	cat $(word 2,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" > $@

genmodel/dundee.%.eventdata:  $(foreach sect,$(DUNDEESECTS),genmodel/dundee.%$(sect).eventdata) | genmodel
	cat $^  >  $@

.PRECIOUS: genmodel/dundee.%.eyedata
genmodel/dundee.%.eyedata: user-dundee-directory.txt $$(shell cat user-dundee-directory.txt)/%ma2p.dat | genmodel
	cat $(word 2,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" > $@

genmodel/dundee.%.eyedata: $(foreach sect,$(DUNDEESECTS),genmodel/dundee.%$(sect).eyedata) | genmodel
	cat $^ > $@

genmodel/dundee.%: $(foreach sect,$(DUNDEESECTS),genmodel/dundee$(sect).%) | genmodel
	cat $^ > $@

.PRECIOUS: genmodel/dundee.%.tokfixns
genmodel/dundee.%.tokfixns: scripts/buildDundeeTokFixns.py genmodel/dundee.textdata genmodel/dundee.%.eyedata genmodel/dundee.%.eventdata \
				scripts/filterLexicon.py genmodel/wsj02to21.wordcounts | genmodel
	python3 $(word 1,$^) $(word 2,$^) $(word 3,$^) $(word 4,$^) | python3 $(word 5,$^) $(word 6,$^) > $@

.PRECIOUS: genmodel/dundee.%_gp.timetoks
genmodel/dundee.%_gp.timetoks: scripts/calcGPtimetoks.py genmodel/dundee.%.tokfixns genmodel/dundee.textdata | genmodel
	python3 $^ > $@

.PRECIOUS: genmodel/dundee.%_fp.timetoks
genmodel/dundee.%_fp.timetoks: scripts/calcFPtimetoks.py genmodel/dundee.%.tokfixns genmodel/dundee.textdata | genmodel
	python3 $^ > $@

.PRECIOUS: genmodel/dundee.%.ngramtimetoks
genmodel/dundee.%.ngramtimetoks: genmodel/dundee.%.timetoks genmodel/dundee.ngramprobtoks | genmodel
	cut -d" " -f2- $(word 2,$^) > $@.tmp
	paste -d" " $< $@.tmp > $@
	rm -rf $@.tmp

.PRECIOUS: %.complextoks
#dundee.wsj02to21-gcg14-3sm-bd.x-efabp.-c_-b2000.complextoks
%.complextoks: %_parsed.output scripts/convert_latin-1.py scripts/spargeComplexity.py scripts/compileComplexity.py genmodel/$$(basename $$(basename $$(basename $$(basename $$(notdir $$@))))).textdata | genmodel
	# This rule is good for dundee data, which is in an odd format and needs to be re-tokenized
	cat $< | python3 $(word 3,$^) > $@.tmp
	python3 $(word 4,$^) $(word 5,$^) $@.tmp > $@
	rm -f $@.tmp

%.complextoks: %_parsed.output scripts/spargeComplexity.py
	# This rule is good for generic, pre-tokenized output
	cat $< | python3 $(word 2,$^) > $@

.PRECIOUS: %.fulltoks
#dundee.wsj02to21-gcg14-3sm-bd.x-efabp.-c_-b2000.fp.fulltoks
%.fulltoks: $$(basename %).complextoks scripts/removeBadToks.py scripts/accumulateMetrics.py scripts/tagPunctuation.py scripts/addLagMetrics.py \
	scripts/padMetrics.py scripts/addLineNums.py $(foreach subj,$(DUNDEESUBJS),genmodel/dundee.$(subj)_$$(subst .,,$$(suffix $$*)).ngramtimetoks) | genmodel
	cut -d" " -f2- $(word 1,$^) > $@.tmp
	$(foreach subj,$(DUNDEESUBJS), cat genmodel/dundee.$(subj)_$(subst .,,$(suffix $*)).ngramtimetoks | python3 $(word 6,$^) | paste -d" " - $@.tmp | python3 $(word 3,$^) | sed 's/-LRB-/(/g;s/-RRB-/)/g;' | python3 $(word 4,$^) >> $@.tmp2;)
	#cat $@.tmp2 | sed 's/-LRB-/(/g;s/-RRB-/)/g;' | python3 $(word 2,$^) | python3 $(word 4,$^) | python3 $(word 5,$^) |
	sed 's/  */ /g;s/F-/fm/g;s/F+/fp/g;s/L-/lm/g;s/L+/lp/g;s/B-/bm/g;s/B+/bp/g;s/Badd/ba/g;s/Bcdr/bc/g;s/BNil/bo/g;s/Bsto/bsto/g;' $@.tmp2 | \
	sed 's/True/1/g;s/False/0/g;' | python3 $(word 7,$^) | python3 $(word 2, $^) > $@
	rm -f $@.tmp{,2}

.PRECIOUS: %.fulltoks.sampled
%.fulltoks.sampled: scripts/sampleComplexity.r %.fulltoks
	$< $(word 2,$^)

.PRECIOUS: %.dundeeeval
#dundee.wsj02to21-gcg14-3sm-bd.x-efabp.-c_-b2000.fp.-fl_-fsc_-fse_-ff_-totsurp.arg.dundeeeval
%.arg.dundeeeval: $$(basename %).fulltoks scripts/dundeeLME.r #| rpkgs
	$(word 2,$^) $< $(subst _, ,$(subst .,,$(suffix $*))) > $@

%.dundeeeval: %.fulltoks scripts/dundeeLME.r #| rpkgs
	$(word 2,$^) $< - > $@
