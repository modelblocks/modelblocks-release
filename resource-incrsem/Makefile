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

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-INCRSEM)

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
	grep 'import' $<  |  sed 's/^from \(.*\) import .*/$$(wildcard $$(RESOURCE-INCRSEM)\/..\/\*\/scripts\/\1.py) \\/;s/^import \(.*\)$$/$$(wildcard $$(RESOURCE-INCRSEM)\/..\/\*\/scripts\/\1.py) \\/'  >>  $@
	echo ''  >>  $@
	cat $@  |  sed 's/\.d:/\.py:/'  >>  $@

#### c++ dependencies
.PRECIOUS:  $(SRC)/%.d
$(SRC)/%.d:  $(SRC)/%.cpp  |  $(INCLUDE)
	echo '$@: \' > $@   #' ##
	echo `g++ -I$| -I$(SRC) -I$(MBDIR)/resource-logreg/include -I$(abspath $(MBDIR)/resource-rvtl/) -std=c++17 -MM $<` | perl -pe 's/^.*: .*?\.cpp */ \$$\(wildcard /;s/\\ *//g;s/$$/\)/' >> $@
	cat $@ | sed 's/\.d:/\.cpp:/' >> $@
endif

#### c++ executables  --  NOTE: $< must precede -l's on ubuntu g++
.PRECIOUS: $(BIN)/%
$(BIN)/%:  $(SRC)/$$(notdir $$@).cpp  $(SRC)/$$(notdir $$@).d  $(CONFIGDIR)/user-cflags.txt  |  $(BIN)  $(INCLUDE)
	g++ -L$(CONDA_PREFIX)/lib -L$(CONDA_PREFIX)/lib64 -I/usr/include/libxml2/ -I$(word 2,$|) -I$(SRC) -I$(MBDIR)/resource-logreg/include -I$(abspath $(MBDIR)/resource-rvtl) -Wall `cat $(word 3,$^)` -fpermissive -std=c++17 $< -lm -larmadillo -Wl,-rpath-link=$(CONDA_PREFIX)/lib -lpthread -o $@

#### copy .ini files to scripts
scripts/%.ini:  $(INCRSEM-SCRIPTS)/%.ini  |  scripts
	cp $^ $@

################################################################################
#
#  1. obtain and pre-process training data
#
################################################################################

## obtain auto-parsed simplewiki corpus...
%simplewikiFULLAUTO.gcg15.linetrees:  /data/corpora/original/english/simplewiki20140903/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-4sm.berk.parsed.linetrees
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
	cat $<  |  grep '^('  |  perl -pe's/[[:^ascii:]]/\!nonasc\!/g'  |  perl $(word 2,$^)  >  $@

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

## remove op and morph tags for eval...
%.nol.senttrees:  %.senttrees
	cat $<  |  perl -pe 's/[-+]l.(?=[^\)]* )//g;s/-x[^-} ][^ |]*[|][^- ]*//g'  >  $@
#	cat $<  |  perl -pe 's/[-+]l.(?=[^\)]* )//g;s/%[^ %|]*\|//g;s/-x(?:(?!-[a-z])[^ }])*(?=[^\)]* )//g'  >  $@
#	cat $<  |  perl -pe 's/[-+]l.(?=[^\)]* )//g;s/-x[^- \}]*%[^- ]*(?=[^\)]* )//g'  >  $@

## note that 'nol' has been changed to 'nol4train'
%.nol4train.linetrees:  %.linetrees
	cat $<  |  perl -pe 's/[-+]l.(?=[^\)]* )//g;s/-x[^-} ][^ |]*[|][^- ]*//g'  >  $@
#	cat $<  |  perl -pe 's/[-+]l.(?=[^\)]* )//g;s/%[^ %|]*\|//g;s/-x(?:(?!-[a-z])[^ }])*(?=[^\)]* )//g'  >  $@
#	cat $<  |  perl -pe 's/[-+]l.(?=[^\)]* )//g;s/-x[^- \}]*%[^- ]*(?=[^\)]* )//g'  >  $@

## remove !ARTICLE article delimiters
%.nodelim.senttrees: %.senttrees
	cat $< | grep -v '!ARTICLE' > $@

## remove scope and coref tags...
%.nosem.linetrees:  %.linetrees
	cat $<  |  perl -pe 's/[-+](?:[nmstw]|nn+)[0-9]+(?=[^\)]* )//g;'  >  $@

## obtain lemma counts from trees...
%.lemmacounts:  %.senttrees  $(INCRSEM-SCRIPTS)/printlemmas.py  $(INCRSEM-SCRIPTS)/printlemmas.d
	cat $<  |  python $(word 2,$^)  |  LC_COLLATE=C sort  |  LC_COLLATE=C uniq -c  |  sed 's/^ *//'  >  $@

## obtain predictors and responses for all semantic processing models...
.PRECIOUS: %sm_semprocdecpars
%sm_semprocdecpars:     $$(basename %).linetrees  \
			$(BIN)/linetrees2sm_semprocdecpars  \
			$$(basename %).lemmacounts  \
			$(INCRSEM-SCRIPTS)/filterFNoncorefPronounsSemprocdecpars.py \
	    #TODO push this filtering further upstream to resource
			#$$(basename %).nol4train.$$(subst .,,$$(suffix $$(subst _,.,$$*)))sm.ccx.model
	#cat $(word 1,$^)  |  grep '^([^)]'  |  grep -v '='  |  $(word 2,$^) $(subst ., ,$(basename $(subst _,.,$(suffix $*)))) $(word 3,$^) $(word 4,$^)  >  $@
	###################################
	### generating training data... ###
	###################################
	cat $(word 1,$^)  |  grep -v '='  |  $(word 2,$^) $(subst ., ,$(basename $(subst _,.,$(suffix $*)))) $(word 3,$^) | python $(word 4,$^) >  $@

## obtain predictors and responses for all semantic processing models...
.PRECIOUS: %densemprocdecpars
%densemprocdecpars:     $$(basename %).linetrees  \
			$(BIN)/linetrees2densemprocdecpars  \
			$$(basename %).lemmacounts  \
			#$$(basename %).nol4train.$$(subst .,,$$(suffix $$(subst _,.,$$*)))sm.ccx.model
	#cat $(word 1,$^)  |  grep '^([^)]'  |  grep -v '='  |  $(word 2,$^) $(subst ., ,$(basename $(subst _,.,$(suffix $*)))) $(word 3,$^) $(word 4,$^)  >  $@
	cat $(word 1,$^)  |  grep -v '='  |  $(word 2,$^) $(subst ., ,$(basename $(subst _,.,$(suffix $*)))) $(word 3,$^) $(word 4,$^) genmodel/test_20dim_2samples_1500epochs0.01lr_nodes_emat_omodel.txt  >  $@

.PRECIOUS: %mlpsemprocdecpars
%mlpsemprocdecpars:     $$(basename %).linetrees  \
			$(BIN)/linetrees2mlpsemprocdecpars  \
			$$(basename %).lemmacounts
	cat $(word 1,$^)  |  grep -v '='  |  $(word 2,$^) $(subst ., ,$(basename $(subst _,.,$(suffix $*)))) $(word 3,$^) $(word 4,$^)  >  $@

.PRECIOUS: %sm_corefsemprocdecpars
%sm_corefsemprocdecpars:     $$(basename %).linetrees  \
			$(BIN)/linetrees2sm_semprocdecpars_coref  \
			$$(basename %).lemmacounts  \
			#$$(basename %).unlinked.nol4train.$$(subst .,,$$(suffix $$(subst _,.,$$*)))sm.ccx.model
	cat $(word 1,$^)  |  grep -v '='  |  $(word 2,$^) $(subst ., ,$(basename $(subst _,.,$(suffix $*)))) $(word 3,$^) $(word 4,$^)  >  $@
#%semprocdecpars:  $$(basename $$(basename %)).linetrees  $(BIN)/linetrees2semprocdecpars  $$(basename $$(basename %)).lemmacounts
#	cat $(word 1,$^)  |  grep '^([^)]'  |  grep -v '='  |  $(word 2,$^) $(subst _, ,$(subst .,, $(suffix $*))) $(word 3,$^)  >  $@
#%f.semprocdecpars:  $$(basename %).linetrees  $(BIN)/linetrees2semprocdecpars  $$(basename %).lemmacounts
#	cat $(word 1,$^)  |  grep '^([^)]'  |  grep -v '='  |  $(word 2,$^) -f$(subst .,,$(suffix $*)) $(word 3,$^)  >  $@
%uc.semprocdecpars:  $$(basename %).linetrees  $(BIN)/linetrees2semprocdecpars  $$(basename %).lemmacounts
	cat $(word 1,$^)  |  grep '^([^)]'  |  grep -v '='  |  $(word 2,$^) -u$(subst .,,$(suffix $*)) $(word 3,$^)  >  $@
%decpars:  $$(basename %).linetrees  $(BIN)/linetrees2$$(subst .,,$$(suffix $$*))decpars  $$(basename %).lemmacounts
	cat $(word 1,$^)  |  grep '^([^)]'  |  grep -v '='  |  $(word 2,$^) $(word 3,$^)  >  $@
#%ss.semprocdecpars:  $$(basename %).linetrees  $(BIN)/linetrees2semprocdecpars  $$(basename %).lemmacounts
#	cat $(word 1,$^)  |  grep '^([^)]'  |  $(word 2,$^) $(word 3,$^) $(subst .,,$(suffix $*))  >  $@

## obtain no-sem predictors and responses ---duan
%.synproccptdecpars:  %.linetrees  $(BIN)/linetrees2synprocdecpars  %.lemmacounts
	cat $(word 1,$^)  |  grep -v '()'  |  $(word 2,$^) $(word 3,$^)  >  $@
%.nod.synproccptdecpars:  %.linetrees  $(BIN)/linetrees2synprocdecpars  %.lemmacounts
	cat $(word 1,$^)  |  grep -v '()'  |  $(word 2,$^) -d $(word 3,$^)  >  $@
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
%.nmlrdecpars:  %semprocdecpars
	cat $<  |  grep '^N'  |  sed 's/^. //'  >  $@
%.fmlrdecpars:  %semprocdecpars 
	cat $<  |  grep '^F'  |  sed 's/^. //'  >  $@
%.jmlrdecpars:  %semprocdecpars
	cat $<  |  grep '^J'  |  sed 's/^. //'  >  $@
%.pcptdecpars:  %semprocdecpars
	cat $<  |  grep '^P'  |  sed 's/^. //'  >  $@
%.acptdecpars:  %semprocdecpars
	cat $<  |  grep '^A'  |  sed 's/^. //'  >  $@
%.bcptdecpars:  %semprocdecpars
	cat $<  |  grep '^B'  |  sed 's/^. //'  >  $@
%.wcptdecpars:  %semprocdecpars
	cat $<  |  grep '^W'  |  sed 's/^. //'  >  $@

## pull apart predictor/responses for training...
%.ngcndecpars:  %densemprocdecpars
	cat $<  |  grep '^N'  |  sed 's/^. //'  >  $@
#%.fmlrdecpars:  %semprocdecpars
%.fgcndecpars:  %densemprocdecpars  $(INCRSEM-SCRIPTS)/filterFNoncorefPronounsSemprocdecpars.py #TODO this won't work - remove and confirm filtering is done earlier
	#cat $<  |  grep '^F'  |  sed 's/^. //'  >  $@
	cat $(word 1, $^)  |  grep '^F'  |  sed 's/^. //' | python $(word 2, $^) >  $@
%.jgcndecpars:  %densemprocdecpars
	cat $<  |  grep '^J'  |  sed 's/^. //'  >  $@
%.pcptdecpars:  %densemprocdecpars
	cat $<  |  grep '^P'  |  sed 's/^. //'  >  $@
%.acptdecpars:  %densemprocdecpars
	cat $<  |  grep '^A'  |  sed 's/^. //'  >  $@
%.bcptdecpars:  %densemprocdecpars
	cat $<  |  grep '^B'  |  sed 's/^. //'  >  $@
%.wcptdecpars:  %densemprocdecpars
	cat $<  |  grep '^W'  |  sed 's/^. //'  >  $@

## pull apart predictor/responses for training...
%.nmlpdecpars:  %mlpsemprocdecpars
	cat $<  |  grep '^N'  |  sed 's/^. //'  >  $@
%.fmlpdecpars:  %mlpsemprocdecpars $(INCRSEM-SCRIPTS)/filterFmlpdecpars_noncorefpronouns.py
	cat $(word 1, $^)  |  grep '^F'  |  sed 's/^. //' | python $(word 2, $^) >  $@
%.jmlpdecpars:  %mlpsemprocdecpars
	cat $<  |  grep '^J'  |  sed 's/^. //'  >  $@
%.pcptdecpars:  %mlpsemprocdecpars
	cat $<  |  grep '^P'  |  sed 's/^. //'  >  $@
%.acptdecpars:  %mlpsemprocdecpars
	cat $<  |  grep '^A'  |  sed 's/^. //'  >  $@
%.bcptdecpars:  %mlpsemprocdecpars
	cat $<  |  grep '^B'  |  sed 's/^. //'  >  $@
%.wcptdecpars:  %mlpsemprocdecpars
	cat $<  |  grep '^W'  |  sed 's/^. //'  >  $@

## push together semantic processing models...
.PRECIOUS: %_semprocmodel
#%semprocmodel:  %.1e-1_1_100_fmlrmodel  %.1e4_1e-6_100_jmlrmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
#%semprocmodel:  %.1e-1_1_100_fmlrmodel  %.1e-1_1_100_jmlrmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
%_semprocmodel: $$(basename %).$$(subst .,_,$$(basename $$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))))_$$(suffix $$(basename $$(basename $$(subst _,.,$$*))))_100_nmlrmodel \
 								$$(basename %).$$(subst .,_,$$(basename $$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))))_$$(suffix $$(basename $$(subst _,.,$$*)))_100_fmlrmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))))_$$(suffix $$(subst _,.,$$*))_100_jmlrmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))))_.pcptmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))))_.acptmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))))_.bcptmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))))_.wcptmodel 
	cat $(word 1,$^)  |  sed 's/^/N /'   >  $@
	cat $(word 2,$^)  |  sed 's/^/F /'  >>  $@
	cat $(word 3,$^)  |  sed 's/^/J /'  >>  $@
	cat $(word 4,$^)  |  sed 's/^/P /'  >>  $@
	cat $(word 5,$^)  |  sed 's/^/A /'  >>  $@
	cat $(word 6,$^)  |  sed 's/^/B /'  >>  $@
	cat $(word 7,$^)  |  sed 's/^/W /'  >>  $@

## push together semantic processing models...
.PRECIOUS: %_corefsemprocmodel
#%semprocmodel:  %.1e-1_1_100_fmlrmodel  %.1e4_1e-6_100_jmlrmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
#%semprocmodel:  %.1e-1_1_100_fmlrmodel  %.1e-1_1_100_jmlrmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
%_corefsemprocmodel: $$(basename %).$$(subst .,_,$$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*))))))_coref$$(suffix $$(basename $$(subst _,.,$$*)))_100_fmlrmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*))))))_coref$$(suffix $$(subst _,.,$$*))_100_jmlrmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*))))))_coref.pcptmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*))))))_coref.acptmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*))))))_coref.bcptmodel \
                $$(basename %).$$(subst .,_,$$(basename $$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*))))))_coref.wcptmodel
	cat $(word 1,$^)  |  sed 's/^/F /'   >  $@
	cat $(word 2,$^)  |  sed 's/^/J /'  >>  $@
	cat $(word 3,$^)  |  sed 's/^/P /'  >>  $@
	cat $(word 4,$^)  |  sed 's/^/A /'  >>  $@
	cat $(word 5,$^)  |  sed 's/^/B /'  >>  $@
	cat $(word 6,$^)  |  sed 's/^/W /'  >>  $@

#.PRECIOUS: %.semproctorchmodel
#%.semproctorchmodel:  %.-batchsize_100_fmlrtorchmodel  %.jmlrtorchmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
#	cat $(word 1,$^)  |  sed 's/^/F /'   >  $@
#	cat $(word 2,$^)  |  sed 's/^/J /'  >>  $@
#	cat $(word 3,$^)  |  sed 's/^/P /'  >>  $@
#	cat $(word 4,$^)  |  sed 's/^/A /'  >>  $@
#	cat $(word 5,$^)  |  sed 's/^/B /'  >>  $@
#	cat $(word 6,$^)  |  sed 's/^/W /'  >>  $@
#
##.PRECIOUS: %.synprocmodel
##%.synprocmodel:  %.1e-1_100_fmlrmodel  %.1e-1_100_jmlrmodel  %.pcptmodel  %.acptmodel  %.bcptmodel  %.wcptmodel
##	cat $(word 1,$^)  |  sed 's/^/F /'   >  $@
##	cat $(word 2,$^)  |  sed 's/^/J /'  >>  $@
##	cat $(word 3,$^)  |  sed 's/^/P /'  >>  $@
##	cat $(word 4,$^)  |  sed 's/^/A /'  >>  $@
##	cat $(word 5,$^)  |  sed 's/^/B /'  >>  $@
##	cat $(word 6,$^)  |  sed 's/^/W /'  >>  $@


################################################################################
#
#  config(ini)-based dsp (densemproc/mlpsemproc)
#
################################################################################

# WS: don't need this bc of VPATH, which is dangerous
# ## Copy learning hyperparameter file from incrsem srcmodel...
# $(PROJECT-SCRIPTS)/%.ini : $(INCRSEM-SCRIPTS)/%.ini  |  $(PROJECT-SCRIPTS)
# 	cp $^ $@

## Induce predicate vectors and cued association transition model (% contains gcn ini filename)...
# wsj requires artdelim for !ARTICLE delimiters
.PRECIOUS: %_emat_omodel
%_emat_omodel : $$(basename %).discgraphs  \
                $(INCRSEM-SCRIPTS)/discgraphs2omodel.py  \
                scripts/$$(subst .,,$$(suffix $$*)).ini
	python3 -u $(word 2,$^) $< $(word 3,$^)  >  $@  2>  $@.log

## Create training example file from training corpus (% contains gcn ini filename and decpar flags)...
.PRECIOUS: %_dspdecpars
%_dspdecpars : $$(basename %).linetrees  \
               $(BIN)/linetrees2densemprocdecpars  \
               $$(basename %)$$(word 1,$$(subst _, ,$$(suffix $$*)))_emat_omodel
	cat $<  |  $(word 2,$^) $(wordlist 2,99,$(subst _, ,$(suffix $*))) $(word 3,$^)  >  $@

## Create training example file from training corpus (% contains decpar flags)...
.PRECIOUS: %_mlpdecpars
%_mlpdecpars : $$(basename %).senttrees  \
               $(BIN)/linetrees2mlpsemprocdecpars  \
               $$(basename %).lemmacounts
	cat $<  |  $(word 2,$^) $(wordlist 1,99,$(subst _, ,$(subst .,,$(suffix $*)))) $(word 3,$^)  >  $@
## remove non-coref pronouns from F predictors (for ontonotes)...
.PRECIOUS: %_ontocorfix_mlpdecpars
%_ontocorfix_mlpdecpars : %_mlpdecpars  $(INCRSEM-SCRIPTS)/filterFmlpdecpars_noncorefpronouns.py
	cat $<  |  python3 $(word 2,$^)  >  $@

## Create component model file from training example file (% contains gcn ini filename and decpar flags and dsp ini filename)...
%_ndspmodel : 
	echo ''  >  $@
%_fdspmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_dspdecpars  \
              $(INCRSEM-SCRIPTS)/dspdecpars2fmodel.py  \
              scripts/$$(subst .,,$$(suffix $$(subst _,., $$*)))f.ini  \
              $$(basename %)$$(word 1,$$(subst _, ,$$(suffix $$*)))_emat_omodel
	cat $<  |  grep '^F'  |  sed 's/^F//'  |  python3 -u $(word 2,$^) $(word 3,$^) $(word 4,$^)  >  $@  2>  $@.log
%_pdspmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_dspdecpars  \
              $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep '^P'  |  python3 $(word 2,$^)  >  $@
%_wdspmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_dspdecpars  \
              $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep '^W'  |  python3 $(word 2,$^)  >  $@
%_jdspmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_dspdecpars  \
              $(INCRSEM-SCRIPTS)/dspdecpars2jmodel.py  \
              scripts/$$(subst .,,$$(suffix $$(subst _,., $$*)))j.ini  \
              $$(basename %)$$(word 1,$$(subst _, ,$$(suffix $$*)))_emat_omodel
	cat $<  |  grep '^J'  |  sed 's/^J//'  |  python3 -u $(word 2,$^) $(word 3,$^) $(word 4,$^)  >  $@  2>  $@.log
%_adspmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_dspdecpars  \
              $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep '^A'  |  python3 $(word 2,$^)  >  $@
%_bdspmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_dspdecpars  \
              $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep '^B'  |  python3 $(word 2,$^)  >  $@

## Create component model file from training example file (% contains decpar flags and mlp ini filename)...
.PRECIOUS: %_nmlpmodel
%_nmlpmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_mlpdecpars  \
              $(INCRSEM-SCRIPTS)/undersample_nmodel_noncorefs.py \
              $(INCRSEM-SCRIPTS)/mlpdecpars2nmodel.py  \
              scripts/$$(subst .,,$$(suffix $$(subst _,., $$*)))n.ini
	cat $<  |  grep '^N'  |  sed 's/^N//'  |  python3 $(word 2,$^) $(word 4,$^) | python3 -u $(word 3,$^) $(word 4,$^)  >  $@  2>  $@.log
.PRECIOUS: %_fmlpmodel
%_fmlpmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_mlpdecpars  \
              $(INCRSEM-SCRIPTS)/mlpdecpars2fmodel.py  \
              scripts/$$(subst .,,$$(suffix $$(subst _,., $$*)))f.ini
	cat $<  |  grep '^F'  |  sed 's/^F //'  |  python3 -u $(word 2,$^) $(word 3,$^)  >  $@  2>  $@.log
.PRECIOUS: %_pmlpmodel
%_pmlpmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_mlpdecpars  \
              $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep '^P'  |  python3 $(word 2,$^)  >  $@
.PRECIOUS: %_wmlpmodel
%_wmlpmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_mlpdecpars  \
              $(INCRSEM-SCRIPTS)/mlpdecpars2wmodel.py  \
              scripts/$$(subst .,,$$(suffix $$(subst _,., $$*)))w.ini
	cat $<  |  grep '^W'  |  sed 's/^W //'  |  python3 -u $(word 2,$^) $(word 3,$^)  >  $@  2>  $@.log
.PRECIOUS: %_jmlpmodel
%_jmlpmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_mlpdecpars  \
              $(INCRSEM-SCRIPTS)/mlpdecpars2jmodel.py  \
              scripts/$$(subst .,,$$(suffix $$(subst _,., $$*)))j.ini
	cat $<  |  grep '^J'  |  sed 's/^J//'  |  python3 -u $(word 2,$^) $(word 3,$^)  >  $@  2>  $@.log
.PRECIOUS: %_amlpmodel
%_amlpmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_mlpdecpars  \
              $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep '^A'  |  python3 $(word 2,$^)  >  $@
.PRECIOUS: %_bmlpmodel
%_bmlpmodel : $$(basename %).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))_mlpdecpars  \
              $(INCRSEM-SCRIPTS)/relfreq.py
	cat $<  |  grep '^B'  |  python3 $(word 2,$^)  >  $@

## Create dsp model out of component models...
.PRECIOUS: %_densemprocmodel
%_densemprocmodel : %_ndspmodel  %_fdspmodel  %_pdspmodel  %_wdspmodel  %_jdspmodel  %_adspmodel  %_bdspmodel  $$(basename %)$$(word 1,$$(subst _, ,$$(suffix $$*)))_emat_omodel
	cat $^  >  $@

## Create mlp model out of component models...
.PRECIOUS: %_mlpsemprocmodel
%_mlpsemprocmodel : %_nmlpmodel  %_fmlpmodel  %_pmlpmodel  %_wmlpmodel  %_jmlpmodel  %_amlpmodel  %_bmlpmodel
	cat $^  >  $@

## Parse with: wsj02.wsj02to21-bestgcn_bestdsp_densemproc-parsed.tokdecs

################################################################################


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
#  syntactic clustering
#
########################################

.PRECIOUS: %lv.linetrees
%lv.linetrees:  $$(basename %).linetrees  $(BIN)/linetrees2lvpcfg  |  $(CONFIGDIR)/user-numthreads.txt
	cat $<  |  grep '('  |  $(word 2,$^) $(subst .,,$(suffix $*)) 1000 $(shell cat $|)  >  $@

########################################
#
#  semantic clustering model
#
########################################

#.PRECIOUS: %_ymodel
#%_ymodel:  $$(basename %).refconts  $(INCRSEM-SCRIPTS)/refconts2reftypes.py
#	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
#	@sleep 5
#	cat $<  |  python $(word 2,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
#
#%typed.linetrees:  $(INCRSEM-SCRIPTS)/make-trees-typed.py  $$(basename %).linetrees  %_2000_5e-2_1e-2_N-aD\:_1_ymodel
#	cat $(word 2,$^)  |  python $(word 1,$^) $(word 3,$^)  >  $@

.PRECIOUS: %hyped.linetrees
%hyped.linetrees:  $$(basename %).linetrees  $(INCRSEM-SCRIPTS)/linetrees2hypedlinetrees.d  $(INCRSEM-SCRIPTS)/linetrees2hypedlinetrees.py
	cat $<  |  OMP_NUM_THREADS=1 python $(word 3,$^) $(subst .,,$(suffix $*))  >  $@  2>  $@.log

########################################
#
#  build PCFG model from rules
#
########################################

%.prtrm.linetrees: %.linetrees      ## also change comma and colon to non-delimiter symbols
	cat $<  |  perl -pe 's/\(([^\(\) ]*) ([^\(\) ]*)\)/\(\1-PRTRM \2\)/g;s/\([^ ]*,[^ -]*/\(Pc/g;s/\(:/\(Pk/g'  >  $@
%.pptrm.linetrees: %.linetrees      ## also change comma and colon to non-delimiter symbols
	cat $<  |  perl -pe 's/\(([^\(\) ]*) ([^\(\) ]*)\)/\(\1-PRTRM \2\)/g;s/\(([^\(\) ]*) (\([^\(\)]*\))\)/\(\1-PRTRM \2\)/g;s/\(([^\(\) ]*) (\([^\(\) ]* \([^\(\)]*\)\))\)/\(\1-PRTRM \2\)/g;s/((?:-l.|-x)*)(-PRTRM)/\2\1/g;s/\([^ ]*,[^ ]* /\(Pc /g;s/\(: /\(Pk /g'  >  $@
%.noprtrm.linetrees: %.linetrees
	cat $< | sed 's/-PRTRM//g' > $@

.PRECIOUS: %pcfg.x-ccu.model
%pcfg.x-ccu.model:  $$(basename %).rules  $(BIN)/pcfgcptdecpars-unker
	cat $<  |  $(word 2,$^) $(subst _, ,$(subst .,,$(suffix $*)))  |  perl -pe 's/CC ([^ ]*) :  =/CC \1 : - - =/;s/CC ([^ ]*) : ([^ ]*) =/Cu \1 : \2 =/'  >  $@
#%.pcfg.x-ccu.model:  %.rules  $(BIN)/pcfgcptdecpars-unker  $(INCRSEM-SCRIPTS)/relfreq.py
#	cat $<  |  $(word 2,$^)  |  python3 $(word 3,$^)  |  perl -pe 's/CC ([^ ]*) :  =/CC \1 : - - =/;s/CC ([^ ]*) : ([^ ]*) =/Cu \1 : \2 =/'  >  $@

########################################
#
#  build model from (e.g. latent-variable) PCFG
#
########################################

%.ccx.model:  %.x-ccu.model            # remove ^g's which may have been added by berkeley trainer, but are safely redundant with -PRTRM
	cat $<  |  perl -pe 's/^Cr : ([^ ]*) =/CC T : \1 T =/;s/^Cu ([^ ]*) : ([^ ]*) =/CC \1 : \2 - =/;s/\^g//g'  >  $@

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
%parsed.semprocdecprobs:  %parsed.tokdecs  $(BIN)/tokdecs2semprocdecprobs  genmodel/$$(basename $$(subst +,-,$$(subst -,.,$$(subst .,,$$(suffix $$*)))))model
	cat $(word 1,$^)  |  $(word 2,$^) $(word 3,$^)  >  $@
##%.semprocdecprobs:  genmodel/$$(basename $$*).semprocdecpars  $(BIN)/semprocdecpars2semprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))model
#%.semprocdecprobs:  genmodel/$$(basename $$(basename $$*)).linetrees  \
#		    $(INCRSEM-SCRIPTS)/unkword.py  genmodel/$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*)))).wordcounts  \
#		    $(BIN)/linetrees2semprocdecpars  genmodel/$$(basename $$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*))))).lemmacounts  \
#		    $(BIN)/semprocdecpars2semprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))model
#	cat $(word 5,$^)  >  eraseme2
#	cat $(word 1,$^)  |  python $(word 2,$^) $(subst unked,,$(subst .,,$(suffix $(basename $*)))) $(word 3,$^)  >  eraseme
#	cat $(word 1,$^)  |  python $(word 2,$^) $(subst unked,,$(subst .,,$(suffix $(basename $*)))) $(word 3,$^)  |  $(word 4,$^) $(word 5,$^)  |  $(word 6,$^) $(word 7,$^)  >  $@

## failed generalization
#%decprobs:  genmodel/$$(basename $$(basename $$*)).$$(subst .,,$$(suffix $$*))decpars  $(BIN)/$$(subst .,,$$(suffix $$*))decpars2$$(subst .,,$$(suffix $$*))decprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$(basename $$*))))model
#	cat $(word 1,$^)  |  $(word 2,$^) $(word 3,$^)  >  $@

%.synprocdecprobs:  genmodel/$$(basename $$*).synproccptdecpars  $(BIN)/synprocdecpars2synprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))model
	cat $(word 1,$^)  |  $(word 2,$^) $(word 3,$^)  >  $@
%.nod.synprocdecprobs:  genmodel/$$(basename $$*).nod.synproccptdecpars  $(BIN)/synprocdecpars2synprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$*)))model
	cat $(word 1,$^)  |  $(word 2,$^) -d $(word 3,$^)  >  $@

%semprocdecprobs:  genmodel/$$(basename $$(basename $$*)).semprocdecpars  $(BIN)/semprocdecpars2semprocdecprobs  genmodel/$$(subst -,.,$$(subst .,,$$(suffix $$(basename $$*))))model
	cat $(word 1,$^)  |  $(word 2,$^) $(subst .,,$(subst _, ,$(suffix $*))) $(word 3,$^)  >  $@


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
%parsed.tokdecs:  bin/$$(subst .,,$$(suffix $$(subst _,.,$$(basename $$(subst +,-,$$(subst -,.,$$*)))))) \
                  $$(basename %).senttoks \
                  genmodel/$$(basename $$(subst +,-,$$(subst -,.,$$(subst .,,$$(suffix $$*)))))model  |  $(CONFIGDIR)/user-numthreads.txt
	###############
	### parsing ###
	###############
	cat $(word 2,$^)  |  $(word 1,$^) -p$(shell cat $|) $(subst _, ,$(subst +,-,$(subst .,,$(suffix $(subst -,.,$*))))) $(word 3,$^)  >  $@  2>  $@.log

### obtain parsed token states for sem proc model...
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
%.senttrees:  %.tokdecs  $(INCRSEM-SCRIPTS)/tokdecs2linetrees.py
	cat $<  |  python $(word 2,$^)  |  perl -pe 's/\[[^\]]*\]://g'  >  $@


