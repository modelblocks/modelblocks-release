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

def ParamVal :=
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

VPATH += $(RESOURCE-ALICE)

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

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-alice-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
ALICEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(ALICEDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(ALICEDIR)$(MSG3))
endif
endif

%/user-alice-directory.txt: | %   
	echo '/home/corpora/original/english/brennan_2016_fmri' > $@

################################################################################
#
#  Reading time data formatting items
#
################################################################################

define 10MMFORM = 
scripts/$(1)%10mm.lmeform: scripts/%10mm.template.lmeform | scripts
	cat $$< | sed 's/ALICEROI/$(1)/g;' > $$@
endef

define ALICE-FORMS 
$(foreach ROI, $(ALICE-ROI), $(eval $(call 10MMFORM,$(ROI))))
endef

$(eval $(ALICE-FORMS))

genmodel/alice.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/alice.rm_unfix_items.params: | genmodel
	echo '-f fdur' > $@

genmodel/alice.spilloverMetrics.params: | genmodel
	echo '-n1 -p fwprob5 totsurp' > $@

genmodel/alice.original.linetoks: srcmodel/alice.original.txt $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt | genmodel
	cat $< | sed "s/‘/\'/g;s/’/\'/g;s/—/ -/g;s/“/\'/g;s/”/\'/g;" | $(word 2, $^) > $@

genmodel/alice.lineitems: srcmodel/alice.original.txt | genmodel
	cat $< | sed "s/‘/\'/g;s/’/\'/g;s/—/ -/g;s/“/\'/g;s/”/\'/g;s/(/-LRB-/g;s/)/-RRB-/g;" > $@

genmodel/alice.src.time.tokmeasures: $(ALICE-SCRIPTS)/textGrid2tokmeasures.py $(ALICEDIR)/alice_stim_shared/DownTheRabbitHoleFinal_exp120_pad_1.TextGrid | genmodel
	cat $(word 2, $^) | python $(word 1, $^) 0 > $@

genmodel/alice.time.tokmeasures: $(ALICE-SCRIPTS)/alignSrcToks2OrigToks.py genmodel/alice.src.time.tokmeasures genmodel/alice.original.linetoks | genmodel
	python $^ >$@

genmodel/alice.linetoks: $(ALICE-SCRIPTS)/tokmeasures2linetoks.py genmodel/alice.time.tokmeasures | genmodel
	cat $(word 2, $^) | python $(word 1, $^) > $@

%alice.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-ALICE)/srcmodel/alice.stripped.linetrees %alice.linetoks
	python $^ > $@

genmodel/alice.sbj.evmeasures: | genmodel
	echo 'subject sampleid docid' > $@
	$(foreach s, $(ALICE-SUBJECTS), for i in {1..362}; do echo "$(s) $$i 1" >> $@; done;)

genmodel/alice.pwr.evmeasures: $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/$(s)-sndpwr.txt) | genmodel
	echo 'sndpwr' > $@
	cat $(wordlist 2, 1000, $^) >> $@

genmodel/alice.rate.evmeasures: $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/$(s)-rate.txt) | genmodel
	echo 'rate' > $@
	cat $(wordlist 2, 1000, $^) >> $@

genmodel/alice.generic.evmeasures: $(CONFIGDIR)/user-alice-directory.txt $(ALICEDIR)/alice_data_shared/sept18-generic-predictors-under-hrf-with-logfrqhal.txt | genmodel/
	echo 'rateConv frqConvOrth breakConvOrth tdpConvOrth tdxConvOrth bupConvOrth buxConvOrth cfgsurpConvOrth bigramlexConvOrth trigramlexConvOrth bigramposConvOrth trigramposConvOrth' > $@.tmp
	tail -n+11 $(word 2, $^) | sed 's/\r//g;' | sed 's/\t/ /g;' >> $@.tmp
	paste -d' ' $@.tmp <(echo 'sampleid'; for i in {1..362}; do echo "$$i"; done;) > $@
	rm -f $@.tmp

genmodel/alice.mvmnt.evmeasures: $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(wildcard $(ALICEDIR)/alice_data_shared/$(s)-rp_*.txt)) | genmodel
	echo 'LATLmvmnt RATLmvmnt LPTLmvmnt LIPLmvmnt LPreMmvmnt LIFGmvmnt' > $@
	$(foreach f, $(wordlist 2, 1000, $^), tail -n+11 $(f) | sed 's/\t/ /g' | sed -e '$$a\' >> $@;)

genmodel/alice.BOLD6mm.evmeasures: $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/6mm/$(s)-timecourses.txt) | genmodel
	echo 'LATL6mm RATL6mm LPTL6mm LIPL6mm LPreM6mm LIFG6mm' > $@
	cat $(wordlist 2, 1000, $^) | sed 's/\t/ /g' >> $@

genmodel/alice.BOLD10mm.evmeasures: $(CONFIGDIR)/user-alice-directory.txt $(foreach s, $(ALICE-SUBJECTS), $(ALICEDIR)/alice_data_shared/10mm/$(s)-timecourses.txt) | genmodel
	echo 'LATL10mm RATL10mm LPTL10mm LIPL10mm LPreM10mm LIFG10mm' > $@
	cat $(wordlist 2, 1000, $^) | sed 's/\t/ /g' >> $@

genmodel/alice.src.evmeasures: genmodel/alice.sbj.evmeasures genmodel/alice.pwr.evmeasures genmodel/alice.rate.evmeasures \
genmodel/alice.mvmnt.evmeasures genmodel/alice.BOLD6mm.evmeasures genmodel/alice.BOLD10mm.evmeasures genmodel/alice.generic.evmeasures $(RESOURCE-RT)/scripts/merge_tables.py | genmodel
	paste -d' ' $(wordlist 1, 6, $^) > $@.tmp
	python $(word 8, $^) $@.tmp $(word 7, $^) sampleid > $@
	rm -f $@.tmp
