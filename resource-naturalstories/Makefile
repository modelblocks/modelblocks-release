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
NaturalStories:
  kernel: SPRCorpus
  text: Natural Stories (self-paced reading corpus)
  value: naturalstories
endef

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

VPATH += $(RESOURCE-NATSTORIES)

NATSTORIES-BASENAME := genmodel/naturalstories.wsj02to21-GRAMMAR-nol-prtrm-3sm-synproc-+c_+u_+b5000.gcg15_nodashtags.merged.syn.5-kenlm.spr
NATSTORIES-LMEDEFAULT := -NSCFl_-o_5_-B_100_-U_3000
$(eval $(call RT-EXPERIMENTS,naturalstories,$(NATSTORIES-BASENAME),$(NATSTORIES-LMEDEFAULT)))

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-naturalstories-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
NATSTORDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(NATSTORDIR))))
RESOURCE-NAME := The Natural Stories Corpus
define RESOURCE-DESCR =

NAME: The Natural Stories Corpus
MB POINTER FILE: config/user-naturalstories-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: A corpus of naturalistic stories meant to contain varied,
low-frequency syntactic constructions. There are a variety of annotations
and psycholinguistic measures available for the stories.
endef
NATSTORIES-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-NATSTORIES = $(error $(NATSTORIES-ERROR))
endif
endif



%/user-naturalstories-directory.txt: | %   
	echo '/home/corpora/original/english/naturalstories' > $@

################################################################################
#
#  Reading time data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(NATSTORIES-SCRIPTS)/%.ini
	cp $< $@

genmodel/naturalstories.merge_tables.params: | genmodel
	echo 'word sentid sentpos' > $@

genmodel/naturalstories.accumulateMetrics.params: | genmodel
	echo 'fwprob5 totsurp -f fdur' > $@

genmodel/naturalstories.spilloverMetrics.params: | genmodel
	echo '-n1 -p fwprob5 totsurp' > $@

genmodel/naturalstories.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/naturalstories.rm_unfix_items.params: | genmodel
	echo '-f fdur' > $@

%/naturalstories.penn.linetrees: $$(DEP-NATSTORIES) $(LTREES-SCRIPTS)/editabletrees2linetrees.pl $(NATSTORDIR)/parses/penn/all-parses.txt.penn \
$(CONFIGDIR)/user-naturalstories-directory.txt | %
	cat $(word 2, $^) | perl $(word 1, $^) > $@

genmodel/naturalstories.linetoks: genmodel/naturalstories.penn.linetrees $(NATSTORIES-SCRIPTS)/penn2sents.py | genmodel
	cat $< | python $(word 2, $^) | sed 's/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g;s/peaked/peeked/g;' > $@

genmodel/naturalstories.delim.linetoks: genmodel/naturalstories.delim.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed | genmodel 
	cat $< | $(word 2, $^) | sed 's/! ARTICLE/!ARTICLE/' | sed 's/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g;s/peaked/peeked/g;' > $@

%naturalstories.items: $$(DEP-NATSTORIES) $(NATSTORDIR)/naturalstories_RTS/processed_RTs.tsv $(RT-SCRIPTS)/get_rttokenization.py \
$(CONFIGDIR)/user-naturalstories-directory.txt
	cat $(word 1, $^) | perl -ne 's/,(?=[A-Z]|[a-z]|[0-9])/ /g;print;' | python $(word, 2, $^) item zone > $@

%/naturalstories.lineitems: $$(DEP-NATSTORIES) %/naturalstories.linetoks $(RT-SCRIPTS)/toks2sents.py \
$(NATSTORDIR)/naturalstories_RTS/all_stories.tok $(CONFIGDIR)/user-naturalstories-directory.txt | %
	cat $(word 3, $^) | sed 's/\t/ /g;s/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g;s/peaked/peeked/g;' | python $(word 2, $^) $(word 1, $^) > $@
 
%/naturalstories.itemmeasures: $$(DEP-NATSTORIES) %/naturalstories.lineitems $(NATSTORIES-SCRIPTS)/docids_by_item.py $(NATSTORDIR)/naturalstories_RTS/all_stories.tok
	cat $(word 1, $^) | python $(word 2, $^) $(word 3, $^) > $@

%naturalstories.mfields.itemmeasures: $$(DEP-NATSTORIES) $(NATSTORDIR)/naturalstories_RTS/all_stories.tok %naturalstories.lineitems \
$(RT-SCRIPTS)/sents2sentids.py $(RT-SCRIPTS)/rename_cols.py $(RT-SCRIPTS)/filter_cols.awk \
$(CONFIGDIR)/user-naturalstories-directory.txt
	paste -d' ' <(cat $(word 1, $^) | sed 's/\t/ /g;s/peaked/peeked/g') <(cat $(word 2, $^) | python $(word 3, $^) | cut -d' ' -f 2-) \
	<(cat $(word 1, $^) | sed 's/\t/ /g;' | awk -f $(word 5, $^) -v cols=item - | python $(word 4, $^) item docid) > $@

naturalstories.%.t.itemmeasures: $$(DEP-NATSTORIES) $(NATSTORDIR)/audio naturalstories.%.itemmeasures $$(NATSTORIES-SCRIPTS)/textGrid2itemmeasures.py
	python $(word 3, $^) $(word 2, $^) $(word 1, $^) > $@

%naturalstories.evmeasures: $$(DEP-NATSTORIES) $(NATSTORDIR)/naturalstories_RTS/processed_RTs.tsv \
$(RT-SCRIPTS)/rename_cols.py $(NATSTORIES-SCRIPTS)/merge_natstor.py %naturalstories.mfields.itemmeasures \
$(DTSR-SCRIPTS)/compute_timestamps_from_RT.py $(CONFIGDIR)/user-naturalstories-directory.txt
	python $(word 3, $^) <(cat $(word 1, $^) | sed 's/\t/ /g;s/peaked/peeked/g;' | python $(word 2, $^) WorkerId subject RT fdur) \
	$(word 4, $^) | sed 's/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g;' | python $(word 5, $^) > $@
  
%naturalstories.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-NATSTORIES)/srcmodel/naturalstories.stripped.linetrees %naturalstories.linetoks
	python $^ > $@

%naturalstories.coref.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-NATSTORIES)/srcmodel/naturalstories.stripped.coref.linetrees %naturalstories.linetoks
	python $^ > $@

