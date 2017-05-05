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

define ParamVal
UCL:
  kernel:
  - ETCorpus
  - SPRCorpus
  text: UCL (eye-tracking)
  value: ucl
endef

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
CONFIG := $(CONFIGDIR)/user-ucl-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
UCLDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(UCLDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(UCLDIR)$(MSG3))
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



%/user-ucl-directory.txt: | %
	echo '/home/corpora/original/english/ucl_novels' > $@



%/user-esn-directory.txt: | %
	echo '/home/corpora/original/english/dundee/echo_state' > $@



%/user-geniatagger-directory.txt: | %
	echo '/home/compling/geniatagger-2.0.2' > $@


################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

genmodel/ucl.merge_tables.params: | genmodel
	echo 'word sentid sentpos' > $@
  
genmodel/ucl.accumulateMetrics.params: | genmodel
	echo 'fwprob5 totsurp' $(ALL-MEM-SPILL) '-f fdurFP' > $@

genmodel/ucl.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/ucl.spilloverMetrics.params: | genmodel
	echo '-n1 -p fwprob5 totsurp' > $@

genmodel/ucl.rm_unfix_items.params: | genmodel
	echo '-f fdurFP' > $@
  
genmodel/ucl.linetoks: $(CONFIGDIR)/user-tokenizer-directory.txt $(CONFIGDIR)/user-ucl-directory.txt \
$(UCLDIR)/eye_movement/stimuli.txt $(TOKENIZERDIR)/ptb_tokenizer.sed | genmodel
	cut -f 2 $(word 3,$^) | tail -n+2 | $(word 4,$^) > $@

genmodel/ucl.lineitems: $(CONFIGDIR)/user-tokenizer-directory.txt $(CONFIGDIR)/user-ucl-directory.txt \
$(UCLDIR)/eye_movement/stimuli.txt | genmodel
	cut -f 2 $(word 3,$^) | tail -n+2 > $@

%ucl.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-UCL)/srcmodel/ucl.stripped.linetrees %ucl.linetoks
	python $^ > $@

genmodel/ucl.src.evmeasures: $(CONFIGDIR)/user-ucl-directory.txt \
$(UCL-SCRIPTS)/build_ucl_corpus.py \
$(UCLDIR)/eye_movement/eyetracking.RT.txt \
$(RT-SCRIPTS)/rename_cols.py \
$(RT-SCRIPTS)/filter_cols.awk | genmodel
	cat $(word 3, $^) | python $(word 2,$^) > $@.tmp
	paste -d' ' $@.tmp <(awk -f $(word 5, $^) -v cols=sentid $@.tmp | python $(word 4, $^) sentid docid) > $@
	rm $@.tmp

.PRECIOUS: $(THISDIR)/esns
$(THISDIR)/esns:
	mkdir $@
  
%.PRECIOUS: esns/ESN.mat
esns/ESN.mat: ESN.zip | esns
	#ESN.zip can be obtained upon request from Stefan Frank
	cp $(word 1,$^) esns/
	unzip esns/ESN.zip -d esns/

.PRECIOUS: genmodel/ucl_postags.mat
genmodel/ucl_postags.mat: $(CONFIGDIR)/user-geniatagger-directory.txt \
$(GENIATAGDIR)/geniatagger genmodel/ucl.linetoks \
$(UCL-SCRIPTS)/build_tagseqs.py esns/ESN.mat
	#requires geniatagger
	ln -s $(shell cat user-geniatagger-directory.txt)/morphdic morphdic
	ln -s $(shell cat user-geniatagger-directory.txt)/models_medline models_medline
	ln -s $(shell cat user-geniatagger-directory.txt)/models_chunking models_chunking
	$(word 2,$^) < $(word 3,$^) | cut -f3 | python $(word 4,$^)
	rm -f morphdic models_medline models_chunking

.PRECIOUS: genmodel/ucl.esnsurps
genmodel/ucl.esnsurps: genmodel/ucl_postags.mat \
$(UCL-SCRIPTS)/do_testESN.m $(UCL-SCRIPTS)/pull_surp.py
	#uses ucl_postags.mat, though they aren't explicitly referenced in the recipe
	matlab -nosplash -nodesktop -r "run('$(basename $(word 2,$^))');quit;"
	python $(word 3,$^) > $@
