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

SHELL := /bin/bash

#### preamble...
.SUFFIXES:
.SECONDEXPANSION:

#### directories used by included items...
BIN := bin
GENMODEL := genmodel

#### included make items...
# include $(dir $(CURDIR))resource-general/Makefile
# include $(RESOURCE-LTREES)/Makefile
# include $(RESOURCE-GCG)/Makefile
# include $(dir $(CURDIR))coref/Makefile
# include $(RESOURCE-LVPCFG)/Makefile
# include $(RESOURCE-LOGREG)/Makefile
# include $(RESOURCE-TREEBANK)/Makefile
# include $(RESOURCE-LCPARSE)/Makefile

#include $(dir $(CURDIR))resource-wordnet/Makefile

INCLUDE := $(abspath $(RESOURCE-INCRSEM)/include)
SRC := $(abspath $(RESOURCE-INCRSEM)/src)

ifndef MAKED
$(info Building source code dependencies using recursive sub-make to $(RESOURCE-INCRSEM)/Makefile...)
## create dependencies for files in python, src, include directories
DUMMY := $(shell $(MAKE) MAKED=1 MAKECONFIG=1 $(subst .py,.d,$(wildcard $(INCRSEM-SCRIPTS)/*.py)) $(subst .cpp,.d,$(wildcard $(SRC)/*.cpp)))
$(info Source code dependencies built and loaded.)
endif

## include those dependencies
include $(wildcard $(RESOURCE-INCRSEM)/*/*.d)

################################################################################
#
#  i. rel freq estimation
#
################################################################################

## relative frequency training for cpt models...
.PRECIOUS: %cptmodel
%cptmodel:  %cptdecpars  $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep -a ' : '  |  python3 $(word 2,$^)  >  $@

.PRECIOUS: %counts
%counts:  %decpars  $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep -a ' : '  |  python3 $(word 2,$^) -n  >  $@


################################################################################
#
#  ii. Code compilation items
#
################################################################################

ifdef MAKED
#### python dependencies
.PRECIOUS:  $(INCRSEM-SCRIPTS)/%.d
$(INCRSEM-SCRIPTS)/%.d:  $(INCRSEM-SCRIPTS)/%.py
	echo '$@: \'  >  $@   #' ##
	grep 'import' $<  |  sed 's/^from \(.*\) import .*/$$(wildcard $$(INCRSEM-SCRIPTS)\/\1.py) \\/;s/^import \(.*\)$$/$$(wildcard $$(INCRSEM-SCRIPTS)\/\1.py) \\/'  >>  $@
	echo ''  >>  $@
	cat $@  |  sed 's/\.d:/\.py:/'  >>  $@

#### c++ dependencies
.PRECIOUS:  $(SRC)/%.d
$(SRC)/%.d:  $(SRC)/%.cpp  |  $(INCLUDE)
	echo '$@: \' > $@   #' ##
	echo `g++ -I$| -I$(MBDIR)/resource-logreg/include -I$(abspath $(MBDIR)/resource-rvtl/) -std=c++11 -MM $<` | sed 's/^.*:.*\.cpp */ $$(wildcard /;s/\\ *//g;s/$$/)/' >> $@
	cat $@ | sed 's/\.d:/\.cpp:/' >> $@
endif

#### c++ executables
.PRECIOUS: $(BIN)/%
$(BIN)/%:  $(SRC)/$$(notdir $$@).cpp  $(SRC)/$$(notdir $$@).d  $(CONFIGDIR)/user-cflags.txt  |  $(BIN)  $(INCLUDE)
	g++ -I/usr/include/libxml2/ -I$(word 2,$|) -I$(MBDIR)/resource-logreg/include -I$(abspath $(MBDIR)/resource-rvtl) -Wall `cat $(word 3,$^)` -fpermissive -std=c++11 -lm -larmadillo -lmlpack -lpthread $< -o $@

################################################################################
#
#  1. obtain and pre-process training data
#
################################################################################

## obtain auto-parsed simplewiki corpus...
%simplewikiFULLAUTO.gcg15.linetrees:  /home/corpora/original/english/simplewiki20140903/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-4sm.berk.parsed.linetrees
	ln -s $< $@

## remove % comments
%.nocomment.linetrees:  %.linetrees
	grep -v '%%%%' $<  >  $@

## remove unparseable characters
%.parseable.linetrees:  %.linetrees  $(INCRSEM-SCRIPTS)/make-trees-parseable.pl
	cat $<  |  grep -v '( *)'  |  grep -v '%%%%'  |  perl $(word 2,$^)  |  sed '/^\s*$$/d'  >  $@

## constrain to just A and M operations...
%.justAM.linetrees:  %.linetrees
	grep -v '\-l[NC]' $<  |  grep -v '([^ ]*-[hgirv]'  >  $@
%.justAMC.linetrees:  %.linetrees
	grep -v '\-l[N]' $<  |  grep -v '([^ ]*-[hgirv]'  >  $@
%.justAMCG.linetrees:  %.linetrees
	cat $<  |  grep -v '([^ ]*-[hirv]'  >  $@
%.justAMCGR.linetrees:  %.linetrees
	cat $<  |  grep -v '([^ ]*-[hiv]'  >  $@

%.qop.linetrees:  %.linetrees  $(INCRSEM-SCRIPTS)/annotateQop.pl
	cat $<  |  perl -pe's/[[:^ascii:]]/\!nonasc\!/g'  |  perl $(word 2,$^)  >  $@

## obtain word counts from trees...
%.wordcounts:  %.linetrees
	cat $<  |  grep -Po '(?<= )[^\(\) \n]+'  |  LC_COLLATE=C sort  |  LC_COLLATE=C uniq -c  >  $@

## get lexicon...
%.lexi:  %.linetrees
	cat $<  |  grep -Po '\([^\(\)\n]+\)'  |  LC_COLLATE=C sort  |  LC_COLLATE=C uniq -c  >  $@

## replace rare words with 'unk'...
%unked.linetrees:  $$(basename %).linetrees  $$(basename %).wordcounts  $(INCRSEM-SCRIPTS)/unkword.py  $(INCRSEM-SCRIPTS)/unkword.d
	cat $<  |  python $(word 3,$^) $(subst .,,$(suffix $*)) $(word 2,$^)  >  $@

## make trees with cuegraph deps as terminals (not currently used)..
%.depterm.linetrees:  %.linetrees  %.cuegraphs  $(INCRSEM-SCRIPTS)/maketreesdepterm.py
	cat $(word 1,$^)  |  python $(word 3,$^) $(word 2,$^)  >  $@

## run left-corner transform on left-headed branches...
%.semlctrans.linetrees:  $(INCRSEM-SCRIPTS)/semlctrans.pl  %.linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@

## run tree-based filler-gap transform...
%.fg.linetrees:  $(GCG-SCRIPTS)/annotateFGTrans.pl  %.linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^) -t  >  $@

## remove op labels for eval...
%.nol.linetrees:  %.linetrees
	cat $<  |  perl -pe 's/[-+]l.(?=[^\)]* )//g;s/-xX[^ \)]* / /g'  >  $@

## obtain lemma counts from trees...
%.lemmacounts:  %.linetrees  $(INCRSEM-SCRIPTS)/printlemmas.py  $(INCRSEM-SCRIPTS)/printlemmas.d
	cat $<  |  python $(word 2,$^)  |  LC_COLLATE=C sort  |  LC_COLLATE=C uniq -c  |  sed 's/^ *//'  >  $@

## obtain predictors and responses for all semantic processing models...
%f.semprocdecpars:  $$(basename %).linetrees  $(BIN)/linetrees2semprocdecpars  $$(basename %).lemmacounts
	cat $(word 1,$^)  |  grep -v '()'  |  grep -v '='  |  $(word 2,$^) -f$(subst .,,$(suffix $*)) $(word 3,$^)  >  $@
%uc.semprocdecpars:  $$(basename %).linetrees  $(BIN)/linetrees2semprocdecpars  $$(basename %).lemmacounts
	cat $(word 1,$^)  |  grep -v '()'  |  grep -v '='  |  $(word 2,$^) -u$(subst .,,$(suffix $*)) $(word 3,$^)  >  $@
%decpars:  $$(basename %).linetrees  $(BIN)/linetrees2$$(subst .,,$$(suffix $$*))decpars  $$(basename %).lemmacounts
	cat $(word 1,$^)  |  grep -v '()'  |  grep -v '='  |  $(word 2,$^) $(word 3,$^)  >  $@
#%ss.semprocdecpars:  $$(basename %).linetrees  $(BIN)/linetrees2semprocdecpars  $$(basename %).lemmacounts
#	cat $(word 1,$^)  |  grep -v '()'  |  $(word 2,$^) $(word 3,$^) $(subst .,,$(suffix $*))  >  $@

## obtain no-sem predictors and responses ---duan
%.synproccptdecpars:  %.linetrees  $(BIN)/linetrees2synprocdecpars  %.lemmacounts
	cat $(word 1,$^)  |  grep -v '()'  |  $(word 2,$^) $(word 3,$^)  >  $@
#%ss.synproccptdecpars:  $$(basename %).linetrees  $(BIN)/linetrees2synprocdecpars  $$(basename %).lemmacounts
#	cat $(word 1,$^)  |  grep -v '()'  |  $(word 2,$^) $(word 3,$^) $(subst .,,$(suffix $*))  >  $@

%.berkunked.synproccptdecpars:  %.synproccptdecpars  $(BIN)/decpars-berkunker
	cat $<  |  $(word 2,$^)  >  $@

## obtain predictors and responses for all semantic processing models...
%.py-semprocdecpars:  %.linetrees  $(INCRSEM-SCRIPTS)/semprocdecpars.py  $(INCRSEM-SCRIPTS)/semprocdecpars.d
	cat $(word 1,$^)  |  grep -v '()'  |  python $(word 2,$^)  >  $@

## convert linetrees to tokdecs
define TargetBlock
Tokdecs:
  blocktitle: Tokdecs (from linetrees)
  targetsuffix: .fromlinetrees.tokdeps
  blockseq:
  - blocktype: Corpus
  - blocktype: TreeOpt
    kwargs:
      outerdelim: .
      innerdelim: .
endef
%.fromlinetrees.tokdecs: %.linetrees  $(INCRSEM-SCRIPTS)/linetrees2tokdecs.py
	cat $(word 1, $^) | python $(word 2,$^) > $@


################################################################################
#
#  2. split predictor-response pairs and merge models
#
################################################################################

## pull apart predictor/responses for training...
%.fmlrdecpars:  %.semprocdecpars
	cat $<  |  grep '^F'  |  sed 's/^. //'  >  $@
%.jmlrdecpars:  %.semprocdecpars
	cat $<  |  grep '^J'  |  sed 's/^. //'  >  $@
%.pcptdecpars:  %.semprocdecpars
	cat $<  |  grep '^P'  |  sed 's/^. //'  >  $@
%.acptdecpars:  %.semprocdecpars
	cat $<  |  grep '^A'  |  sed 's/^. //'  >  $@
%.bcptdecpars:  %.semprocdecpars
	cat $<  |  grep '^B'  |  sed 's/^. //'  >  $@
%.wcptdecpars:  %.semprocdecpars
	cat $<  |  grep '^W'  |  sed 's/^. //'  >  $@

## push together semantic processing models...
.PRECIOUS: %.semprocmodel
%.semprocmodel:  %.1e-1_1_100_fmlrmodel  %.1000_1e-2_100_jmlrmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
	cat $(word 1,$^)  |  sed 's/^/F /'   >  $@
	cat $(word 2,$^)  |  sed 's/^/J /'  >>  $@
	cat $(word 3,$^)  |  sed 's/^/P /'  >>  $@
	cat $(word 4,$^)  |  sed 's/^/A /'  >>  $@
	cat $(word 5,$^)  |  sed 's/^/B /'  >>  $@
	cat $(word 6,$^)  |  sed 's/^/W /'  >>  $@
.PRECIOUS: %.semproctorchmodel
%.semproctorchmodel:  %.-batchsize_100_fmlrtorchmodel  %.jmlrtorchmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
	cat $(word 1,$^)  |  sed 's/^/F /'   >  $@
	cat $(word 2,$^)  |  sed 's/^/J /'  >>  $@
	cat $(word 3,$^)  |  sed 's/^/P /'  >>  $@
	cat $(word 4,$^)  |  sed 's/^/A /'  >>  $@
	cat $(word 5,$^)  |  sed 's/^/B /'  >>  $@
	cat $(word 6,$^)  |  sed 's/^/W /'  >>  $@
#.PRECIOUS: %.synprocmodel
#%.synprocmodel:  %.1e-1_100_fmlrmodel  %.1e-1_100_jmlrmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
#	cat $(word 1,$^)  |  sed 's/^/F /'   >  $@
#	cat $(word 2,$^)  |  sed 's/^/J /'  >>  $@
#	cat $(word 3,$^)  |  sed 's/^/P /'  >>  $@
#	cat $(word 4,$^)  |  sed 's/^/A /'  >>  $@
#	cat $(word 5,$^)  |  sed 's/^/B /'  >>  $@
#	cat $(word 6,$^)  |  sed 's/^/W /'  >>  $@

### Get training and test data

genmodel/simplewikiFULLAUTO.gcg15.morphed.1000unked.justAM.unlinked.train.semlctrans.fg.linetrees: genmodel/simplewikiFULLAUTO.gcg15.morphed.1000unked.justAM.unlinked.semlctrans.fg.linetrees
	split -l 300000 $< genmodel/train; mv genmodel/trainaa $@; head -n 100000 genmodel/trainab > genmodel/simplewikiFULLAUTO.gcg15.morphed.1000unked.justAM.unlinked.test.semlctrans.fg.linetrees

########################################
#
#  build model directly from rel freq estim on corpus
#
########################################

.PRECIOUS: %.synprocmodel
%.synprocmodel:  %.synproccptmodel
	cp $^ $@

########################################
#
#  semantic clustering model
#
########################################

.PRECIOUS: %_ymodel
%_ymodel:  $$(basename %).refconts  $(INCRSEM-SCRIPTS)/refconts2reftypes.py
	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
	@sleep 5
	cat $<  |  python $(word 2,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@

%typed.linetrees:  $(INCRSEM-SCRIPTS)/make-trees-typed.py  $$(basename %).linetrees  %_2000_5e-2_1e-2_N-aD\:_1_ymodel
	cat $(word 2,$^)  |  python $(word 1,$^) $(word 3,$^)  >  $@

########################################
#
#  build model from simple (binary) PCFG
#
########################################

.PRECIOUS: %.pcfg.x-ccu.model
%.pcfg.x-ccu.model:  %.rules  $(BIN)/pcfgcptdecpars-unker
	cat $<  |  $(word 2,$^)  |  perl -pe 's/CC ([^ ]*) :  =/CC \1 : - - =/;s/CC ([^ ]*) : ([^ ]*) =/Cu \1 : \2 =/'  >  $@
#%.pcfg.x-ccu.model:  %.rules  $(BIN)/pcfgcptdecpars-unker  $(INCRSEM-SCRIPTS)/relfreq.py
#	cat $<  |  $(word 2,$^)  |  python3 $(word 3,$^)  |  perl -pe 's/CC ([^ ]*) :  =/CC \1 : - - =/;s/CC ([^ ]*) : ([^ ]*) =/Cu \1 : \2 =/'  >  $@

%.prtrm.linetrees: %.linetrees
	cat $<  |  perl -pe 's/\(([^\(\) ]*) ([^\(\) ]*)\)/\(\1-PRTRM \2\)/g'  >  $@

########################################
#
#  build model from latent-variable PCFG
#
########################################

.PRECIOUS: %.synprocmodel
%.synprocmodel:  %.x-cc.model  $(BIN)/ccmodel2synproccptmodel
	cat $<  |  sed 's/^Cr/R/'  |  $(word 2,$^)  >  $@

.PRECIOUS: %.fjabpmodel
%.fjabpmodel:  %.x-cc.model  $(BIN)/ccmodel2fjabpcptmodel
	cat $<  |  sed 's/^Cr/R/'  |  $(word 2,$^)  >  $@

########################################
#
#  trace of decisions and model probs
#
########################################

##%parsed.semprocdecprobs:  %parsed.linetrees  $(BIN)/linetrees2semprocdecpars  genmodel/$$(basename $$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*))))).lemmacounts  $(BIN)/semprocdecpars2semprocdecprobs  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*))))model
##	cat $(word 1,$^)  |  $(word 2,$^) $(word 3,$^)  |  $(word 4,$^) $(word 5,$^)  >  $@
%parsed.semprocdecprobs:  %parsed.tokdecs  $(BIN)/tokdecs2semprocdecprobs  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*))))model
	cat $(word 1,$^)  |  $(word 2,$^) $(word 3,$^)  >  $@
##%.semprocdecprobs:  genmodel/$$(basename $$*).semprocdecpars  $(BIN)/semprocdecpars2semprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))model
#%.semprocdecprobs:  genmodel/$$(basename $$(basename $$*)).linetrees  \
#		    $(INCRSEM-SCRIPTS)/unkword.py  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*)))).wordcounts  \
#		    $(BIN)/linetrees2semprocdecpars  genmodel/$$(basename $$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*))))).lemmacounts  \
#		    $(BIN)/semprocdecpars2semprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))model
#	cat $(word 5,$^)  >  eraseme2
#	cat $(word 1,$^)  |  python $(word 2,$^) $(subst unked,,$(subst .,,$(suffix $(basename $*)))) $(word 3,$^)  >  eraseme
#	cat $(word 1,$^)  |  python $(word 2,$^) $(subst unked,,$(subst .,,$(suffix $(basename $*)))) $(word 3,$^)  |  $(word 4,$^) $(word 5,$^)  |  $(word 6,$^) $(word 7,$^)  >  $@
%.semprocdecprobs:  genmodel/$$(basename $$*).semprocdecpars  $(BIN)/semprocdecpars2semprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))model
	cat $(word 1,$^)  |  $(word 2,$^) $(word 3,$^)  >  $@


#################################################
#
#  training F model with mallet package
#
##############################################

%.malletInput:  %.fmlrdecpars $(INCRSEM-SCRIPTS)/getMalletInput.py
	cat $(word 1,$^)  |  python $(word 2,$^)  >  $@

%.mallet: %.malletInput
	mallet-2.0.8RC2/bin/mallet import-file --input $< --output $@

%.mallet.classifier: %.mallet
	mallet-2.0.8RC2/bin/mallet train-classifier --input $< --output-classifier $@ --trainer MaxEnt


%.test.malletOutput: %.test.malletInput %.train.mallet.classifier
	mallet-2.0.8RC2/bin/mallet classify-file --input $(word 1, $^) --output $@ --classifier $(word 2, $^) 


#################################################
#
# training F model with MegaM package
#
##############################################

#currently magem does not run on dignam
%.megam:  %.fmlrdecpars $(INCRSEM-SCRIPTS)/getMegamInput.py
	cat $(word 1,$^)  |  python $(word 2,$^) >  $@

#this needs to be run on emilia
%.megam.classifier: %.megam
	/scratch/libs/megam_0.92/megam_i686.opt multiclass $< > $@


#################################################
#
# training F model with Vowpal Wabbit  package
#
##############################################

#VW needs the label to be integers from 1 to k

FmodelLabelIndexUnked.pkl: $(INCRSEM-SCRIPTS)/getFlabelIndex.py genmodel/simplewikiFULLAUTO.gcg15.morphed.1000unked.justAM.unlinked.fmlrdecpars
	cat $(word 2, $^) | python $(word 1,$^) $@

%.VW.input: %.fmlrdecpars $(INCRSEM-SCRIPTS)/getVWinput.py FmodelLabelIndexUnked.pkl
	cat $(word 1,$^)  |  python $(word 2,$^) $(word 3,$^) >  $@

%.VW.classifier: %.VW.input
	vowpal_wabbit/vowpalwabbit/vw --oaa 286345 $< -f $@

%.VW.predict: genmodel/simplewikiFULLAUTO.gcg15.morphed.1000unked.justAM.unlinked.train.VW.classifier %.VW.input
	vowpal_wabbit/vowpalwabbit/vw -t -i $(word 1, $^) $(word 2, $^) -p $@


################################################################################
#
#  3. parse
#
################################################################################

## OBSOLETE: obtain parse output for sem proc model...
%semproc-py-parsed.output:  $(INCRSEM-SCRIPTS)/semproc.py  $$(basename %).linetoks  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*)))).semprocmodel
	cat $(word 2,$^)  |  python $(word 1,$^) $(word 3,$^)  >  $@

define ParamVal
Synproc:
  kernel: Parser
  text: Incremental syntactic processing parser (synproc)
  value: synproc
endef
define TargetBlock
ParseTokdecs:
  blocktitle: Parse (Tokdecs)
  targetsuffix: _parsed.tokdecs
  family: Parse
  blockseq:
  - blocktype: Corpus
    kwargs:
      instancename: Test
  - blocktype: String
    value: .
  - blocktype: ParseParams
    kwargs:
      innerdelim: '-'
endef

.PRECIOUS: %parsed.tokdecs
%parsed.tokdecs:  bin/$$(subst .,,$$(suffix $$(basename $$(subst +,-,$$(subst -,.,$$*))))) \
                  $$(basename %).linetoks \
                  genmodel/$$(basename $$(subst +,-,$$(subst -,.,$$(subst .,,$$(suffix $$*)))))model
	cat $(word 2,$^)  |  $(word 1,$^) $(subst _, ,$(subst +,-,$(subst .,,$(suffix $(subst -,.,$*))))) $(word 3,$^)  >  $@

### obtain parsed token states for sem proc model...
##.PRECIOUS: %semproc-parsed.tokdecs
##%semproc-parsed.tokdecs:  bin/semproc  $$(basename %).linetoks  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*)))).semprocmodel
##	cat $(word 2,$^)  |  $(word 1,$^) $(word 3,$^)  >  $@
##.PRECIOUS: %ss-semproc-parsed.tokdecs
##%ss-semproc-parsed.tokdecs:  bin/semproc  $$(basename %).linetoks  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))ss.semprocmodel
##	cat $(word 2,$^)  |  $(word 1,$^) $(subst .,,$(suffix $(subst -,.,$*))) $(word 3,$^)  >  $@
##.PRECIOUS: %semproctorch-parsed.tokdecs
##%semproctorch-parsed.tokdecs:  bin/semproc  $$(basename %).linetoks  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*)))).semproctorchmodel
##	cat $(word 2,$^)  |  $(word 1,$^) $(word 3,$^)  >  $@
##.PRECIOUS: %ss-semproctorch-parsed.tokdecs
##%ss-semproctorch-parsed.tokdecs:  bin/semproc  $$(basename %).linetoks  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))ss.semproctorchmodel
##	cat $(word 2,$^)  |  $(word 1,$^) $(subst .,,$(suffix $(subst -,.,$*))) $(word 3,$^)  >  $@

### added to force using synproc --duan
#.PRECIOUS: %synproc-parsed.tokdecs
#%synproc-parsed.tokdecs:  bin/synproc  $$(basename %).linetoks  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*)))).synproccptmodel
#	cat $(word 2,$^)  |  $(word 1,$^) $(word 3,$^)  >  $@

##.PRECIOUS: %ss-synproc-parsed.tokdecs
##%ss-synproc-parsed.tokdecs:  bin/synproc  $$(basename %).linetoks  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))ss.synproccptmodel
##	cat $(word 2,$^)  |  $(word 1,$^) $(subst .,,$(suffix $(subst -,.,$*))) $(word 3,$^)  >  $@

## obtain linetrees from token decisions...
%.linetrees:  %.tokdecs  $(INCRSEM-SCRIPTS)/tokdecs2linetrees.py
	cat $<  |  python $(word 2,$^)  |  perl -pe 's/\[[^\]]*\]://g'  >  $@

