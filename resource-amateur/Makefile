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
#  include $(RESOURCE-TOKENIZER)/Makefile
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
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

SPILL-DUNDEE := S1 S2 S3
DLT-DUNDEE := dlt dltc dltcv dltv dltm dltcm dltcvm dltvm
FJ-DUNDEE := fjprob endembd embdlen embdlendr embdlendrv noF noFlen noFlendr noFlendrv reinst reinstlen reinstlendr reinstlendrv
ALL-MEM = $(DLT-DUNDEE) $(FJ-DUNDEE)
ALL-MEM-SPILL = $(ALL-MEM) $(foreach var,$(ALL-MEM),$(foreach spill,$(SPILL-DUNDEE),$(var)$(spill)))

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
CONFIG := $(CONFIGDIR)/user-amateur-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
AMATEURDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(AMATEURDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(AMATEURDIR)$(MSG3))
endif

CONFIG := $(CONFIGDIR)/user-esn-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
ESNDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(ESNDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(ESNDIR)$(MSG3))
endif

CONFIG := $(CONFIGDIR)/user-geniatagger-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GENIATAGDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GENIATAGDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(GENIATAGDIR)$(MSG3))
endif
endif



%/user-amateur-directory.txt: | %
	echo '/home/corpora/original/english/amateur_novels' > $@



%/user-esn-directory.txt: | %
	echo '/home/corpora/original/english/dundee/echo_state' > $@



%/user-geniatagger-directory.txt: | %
	echo '/home/compling/geniatagger-2.0.2' > $@


################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

genmodel/amateur.merge_tables.params: | genmodel
	echo 'word sentid sentpos' > $@
  
genmodel/amateur.accumulateMetrics.params: | genmodel
	echo 'fwprob5 totsurp' $(ALL-MEM-SPILL) '-f fdurFP' > $@

genmodel/amateur.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/amateur.rm_unfix_items.params: | genmodel
	echo '-f fdurFP' > $@
  
genmodel/amateur.linetoks: $(CONFIGDIR)/user-tokenizer-directory.txt $(CONFIGDIR)/user-amateur-directory.txt \
$(AMATEURDIR)/eye_movement/stimuli.txt $(TOKENIZERDIR)/ptb_tokenizer.sed | genmodel
	cut -f 2 $(word 3,$^) | tail -n+2 | $(word 4,$^) > $@

genmodel/amateur.lineitems: $(CONFIGDIR)/user-tokenizer-directory.txt $(CONFIGDIR)/user-amateur-directory.txt \
$(AMATEURDIR)/eye_movement/stimuli.txt | genmodel
	cut -f 2 $(word 3,$^) | tail -n+2 > $@

%amateur.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-AMATEUR)/srcmodel/amateur.stripped.linetrees %amateur.linetoks
	python $^ > $@

# The *.fp.* component is vacuous here since these data contain both FP and GP.
# It's just there to make the %.itemmeasures generation work downstream.
genmodel/amateur.fp.src.evmeasures: $(CONFIGDIR)/user-amateur-directory.txt \
$(AMATEUR-SCRIPTS)/build_amateur_corpus.py \
$(AMATEURDIR)/eye_movement/eyetracking.RT.txt | genmodel
	cat $(word 3, $^) | python $(word 2,$^) > $@

.PRECIOUS: $(THISDIR)/esns
$(THISDIR)/esns:
	mkdir $@
  
%.PRECIOUS: esns/ESN.mat
esns/ESN.mat: ESN.zip | esns
	#ESN.zip can be obtained upon request from Stefan Frank
	cp $(word 1,$^) esns/
	unzip esns/ESN.zip -d esns/

.PRECIOUS: genmodel/amateur_postags.mat
genmodel/amateur_postags.mat: $(CONFIGDIR)/user-geniatagger-directory.txt \
$(GENIATAGDIR)/geniatagger genmodel/amateur.linetoks \
$(AMATEUR-SCRIPTS)/build_tagseqs.py esns/ESN.mat
	#requires geniatagger
	ln -s $(shell cat user-geniatagger-directory.txt)/morphdic morphdic
	ln -s $(shell cat user-geniatagger-directory.txt)/models_medline models_medline
	ln -s $(shell cat user-geniatagger-directory.txt)/models_chunking models_chunking
	$(word 2,$^) < $(word 3,$^) | cut -f3 | python $(word 4,$^)
	rm -f morphdic models_medline models_chunking

.PRECIOUS: genmodel/amateur.esnsurps
genmodel/amateur.esnsurps: genmodel/amateur_postags.mat \
$(AMATEUR-SCRIPTS)/do_testESN.m $(AMATEUR-SCRIPTS)/pull_surp.py
	#uses amateur_postags.mat, though they aren't explicitly referenced in the recipe
	matlab -nosplash -nodesktop -r "run('$(basename $(word 2,$^))');quit;"
	python $(word 3,$^) > $@
