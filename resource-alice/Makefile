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
#  Includes
#
#  Include statements to this file should be preceded by the following includes
#
#  include $(dir $(CURDIR))resource-general/Makefile
#  include $(RESOURCE-LTREES)/Makefile
#  include $(RESOURCE-GCG)/Makefile
#  include $(RESOURCE-RT)/Makefile
#  include $(RESOURCE-LVPCFG)/Makefile
#  include $(RESOURCE-LCPARSE)/Makefile
#  include $(RESOURCE-BNC)/Makefile
#  include $(RESOURCE-TREEBANK)/Makefile
#  include $(RESOURCE-GIGAWORD)/Makefile
#  include $(RESOURCE-KENLM)/Makefile
#  include $(RESOURCE-SRILM)/Makefile
#  include $(RESOURCE-INCRSEM)/Makefile
#
################################################################################

################################################################################
#
#  MB-DOC 
#
################################################################################

define ParamVal
Alice:
  kernel: fMRICorpus
  text: Alice in Wonderland, Chapter 1
  value: alice
endef

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-ALICE)

SAMPLE-START := 20.05891632653599
# NOTE: Dataset also contains an s33 but BOLD values are all zero, so excluded from analysis
ALICE-SUBJECTS := s18 s22 s23 s24 s26 s28 s30 s31 s35 s36 s37 s38 s39 s40 s41 s42 s43 s44 s45 s46 s47 s48 s49 s50 s51 s52 s53
ALICE-ROI := LATL LIFG LIPL LPreM LPreM LPTL RATL

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-alice-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
ALICEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(ALICEDIR))))
RESOURCE-NAME := Alice in Wonderland Corpus (Brennan et al 16)
define RESOURCE-DESCR =

NAME: The Alice in Wonderland corpus
MB POINTER FILE: config/user-alice-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: fMRI data from 28 subjects listening to the first chapter of Alice in Wonderland.
Collected by Brennan et al (2016).
endef
ALICE-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-ALICE = $(error $(ALICE-ERROR))
endif
endif

%/user-alice-directory.txt: | %   
	echo '/fs/project/lin-dept/corpora/original/english/brennan_2016_fmri' > $@

################################################################################
#
#  Data formatting items
#
################################################################################

%/alice.original.linetoks: $$(DEP-TOKENIZER) srcmodel/alice.original.txt $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt | %
	cat $< | sed "s/‘/\'/g;s/’/\'/g;s/—/ -/g;s/“/\'/g;s/”/\'/g;" | $(word 2, $^) > $@

genmodel/alice.lineitems: srcmodel/alice.original.txt | genmodel
	cat $< | sed "s/‘/\'/g;s/’/\'/g;s/—/ -/g;s/“/\'/g;s/”/\'/g;s/(/-LRB-/g;s/)/-RRB-/g;" > $@

%/alice.src.time.tokmeasures: $$(DEP-ALICE) $(ALICE-SCRIPTS)/textGrid2tokmeasures.py $(ALICEDIR)/alice_stim_shared/DownTheRabbitHoleFinal_exp120_pad_1.TextGrid | %
	cat $(word 2, $^) | python $(word 1, $^) 0 > $@

genmodel/alice.time.tokmeasures: $(ALICE-SCRIPTS)/alignSrcToks2OrigToks.py genmodel/alice.src.time.tokmeasures genmodel/alice.original.linetoks | genmodel
	python $^ >$@

genmodel/alice.linetoks: $(ALICE-SCRIPTS)/tokmeasures2linetoks.py genmodel/alice.time.tokmeasures | genmodel
	cat $(word 2, $^) | python $(word 1, $^) > $@

%alice.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-ALICE)/srcmodel/alice.stripped.linetrees %alice.linetoks
	python $^ > $@

genmodel/alice.sbj.evmeasures: | genmodel
	echo 'subject sampleid docid time' > $@
	$(foreach s, $(ALICE-SUBJECTS), for i in {1..362}; do echo "$(s) $$i 1 $$((($$i-1)*2))" >> $@; done;)

%/alice.pwr.evmeasures: $$(DEP-ALICE) $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/$(s)-sndpwr.txt) | %
	echo 'sndpwr' > $@
	cat $(wordlist 2, 1000, $^) >> $@

%/alice.rate.evmeasures: $$(DEP-ALICE) $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/$(s)-rate.txt) | %
	echo 'Rate' > $@
	cat $(wordlist 2, 1000, $^) >> $@

%/alice.generic.evmeasures: $$(DEP-ALICE) $(CONFIGDIR)/user-alice-directory.txt $(ALICEDIR)/alice_data_shared/sept18-generic-predictors-under-hrf-with-logfrqhal.txt | %
	echo 'RateConv frqConvOrth breakConvOrth tdpConvOrth tdxConvOrth bupConvOrth buxConvOrth cfgsurpConvOrth bigramlexConvOrth trigramlexConvOrth bigramposConvOrth trigramposConvOrth' > $@.tmp
	tail -n+11 $(word 2, $^) | sed 's/\r//g;' | sed 's/\t/ /g;' >> $@.tmp
	paste -d' ' $@.tmp <(echo 'sampleid'; for i in {1..362}; do echo "$$i"; done;) > $@
	rm -f $@.tmp

%/alice.mvmnt.evmeasures: $$(DEP-ALICE) $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(wildcard $(ALICEDIR)/alice_data_shared/$(s)-rp_*.txt)) | %
	echo 'LATLmvmnt RATLmvmnt LPTLmvmnt LIPLmvmnt LPreMmvmnt LIFGmvmnt' > $@
	$(foreach f, $(wordlist 2, 1000, $^), tail -n+11 $(f) | sed 's/^ *//g' | sed 's/\t/ /g' | sed 's/ \+/ /g' | sed -e '$$a\' >> $@;)

%/alice.BOLD6mm.evmeasures: $$(DEP-ALICE) $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/6mm/$(s)-timecourses.txt) | genmodel
	echo 'boldLATL6mm boldRATL6mm boldLPTL6mm boldLIPL6mm boldLPreM6mm boldLIFG6mm' > $@
	cat $(wordlist 2, 1000, $^) | sed 's/^ *//g' | sed 's/\t/ /g' | sed 's/ \+/ /g' >> $@

%/alice.BOLD10mm.evmeasures: $$(DEP-ALICE) $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/10mm/$(s)-timecourses.txt) | genmodel
	echo 'boldLATL10mm boldRATL10mm boldLPTL10mm boldLIPL10mm boldLPreM10mm boldLIFG10mm' > $@
	cat $(wordlist 2, 1000, $^) | sed 's/^ *//g' | sed 's/\t/ /g' | sed 's/ \+/ /g' >> $@

%/alice.evmeasures: %/alice.sbj.evmeasures %/alice.pwr.evmeasures %/alice.rate.evmeasures \
%/alice.mvmnt.evmeasures %/alice.BOLD6mm.evmeasures %/alice.BOLD10mm.evmeasures %/alice.generic.evmeasures $(RESOURCE-RT)/scripts/merge_tables.py | %
	paste -d' ' $(wordlist 1, 6, $^) > $@.tmp
	python $(word 8, $^) $@.tmp $(word 7, $^) sampleid > $@
	rm -f $@.tmp

%/alice.docids.itemmeasures: %/alice.lineitems $(ALICE-SCRIPTS)/docids_by_item.py
	cat $< | python $(word 2, $^) > $@

alice.%.t.itemmeasures: genmodel/alice.time.tokmeasures alice.%.itemmeasures $(ALICE-SCRIPTS)/timestamps_by_item.py
	cat $< | python $(word 3, $^) $(word 2, $^) > $@

