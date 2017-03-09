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
CONFIG := $(CONFIGDIR)/user-naturalstories-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
NATSTORDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(NATSTORDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(NATSTORDIR)$(MSG3))
endif
endif



%/user-naturalstories-directory.txt: | %   
	echo '/home/corpora/original/english/naturalstories' > $@

################################################################################
#
#  Reading time data formatting items
#
################################################################################

# DELETE once we can move past the buggy natstories dev/test split
%.iz.ngram.itemmeasures: %.ngram.itemmeasures $(NATSTORDIR)/naturalstories_RTS/all_stories.tok \
../resource-rt/scripts/filter_cols.py $(CONFIGDIR)/user-naturalstories-directory.txt | genmodel
	paste -d' ' $< <(cat $(word 2, $^) | sed 's/\t/ /g' | python $(word 3, $^) -d -c item zone) > $@

genmodel/naturalstories.merge_tables.params: | genmodel
	echo 'word sentid sentpos' > $@

genmodel/naturalstories.accumulateMetrics.params: | genmodel
	echo 'fwprob5 totsurp -f fdur' > $@

genmodel/naturalstories.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/naturalstories.rm_unfix_items.params: | genmodel
	echo '-f fdur' > $@

genmodel/naturalstories.linetrees: $(LTREES-SCRIPTS)/editabletrees2linetrees.pl $(NATSTORDIR)/parses/penn/all-parses.txt.penn \
$(CONFIGDIR)/user-naturalstories-directory.txt | genmodel
	cat $(word 2, $^) | perl $< > $@

genmodel/naturalstories.linetoks: genmodel/naturalstories.linetrees $(NATSTORIES-SCRIPTS)/penn2sents.py | genmodel
	cat $< | python $(word 2, $^) | sed 's/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g;' > $@

genmodel/naturalstories.items: $(NATSTORDIR)/naturalstories_RTS/processed_RTs.csv $(RT-SCRIPTS)/get_rttokenization.py \
$(CONFIGDIR)/user-naturalstories-directory.txt
	cat $< | perl -ne 's/,(?=[A-Z]|[a-z]|[0-9])/ /g;print;' | python $(word, 2, $^) item zone > $@

genmodel/naturalstories.lineitems: genmodel/naturalstories.linetoks $(RT-SCRIPTS)/toks2sents.py \
$(NATSTORDIR)/naturalstories_RTS/all_stories.tok $(CONFIGDIR)/user-naturalstories-directory.txt | genmodel
	cat $(word 3, $^) | sed 's/\t/ /g;s/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g;' | python $(word 2, $^) $< > $@
  
genmodel/naturalstories.mfields.itemmeasures: $(NATSTORDIR)/naturalstories_RTS/all_stories.tok genmodel/naturalstories.lineitems \
$(RT-SCRIPTS)/sents2sentids.py $(CONFIGDIR)/user-naturalstories-directory.txt | genmodel
	paste -d' ' <(cat $< | sed 's/\t/ /g;') <(cat $(word 2, $^) | python $(word 3, $^) | cut -d' ' -f 2-) > $@

# Just here to preserve old buggy dev/test split
genmodel/naturalstories.mfields.o.itemmeasures: genmodel/naturalstories.mfields.itemmeasures \
$(RT-SCRIPTS)/roll_toks.py genmodel/naturalstories.gold.dlt.tokmeasures $(RT-SCRIPTS)/filter_cols.py \
$(CONFIGDIR)/user-naturalstories-directory.txt | genmodel
	paste -d' ' <(cat $< | python $(word 4, $^) -x sentid) <(cat $(word 3, $^) | sed 's/``/'\''/g;s/'\'\''/'\''/g;s/-LRB-/(/g;s/-RRB-/)/g;' | \
  python $(word 2, $^) $< -e latin-1 | python $(word 4, $^) -c sentid) > $@
  
# The *.fp.* component is vacuous here since these are SPR data. It's just there to make the RT toks generation work.
genmodel/naturalstories.fp.src.evmeasures: $(NATSTORDIR)/naturalstories_RTS/processed_RTs.csv \
$(RT-SCRIPTS)/rename_cols.py $(RT-SCRIPTS)/merge_tables.py genmodel/naturalstories.mfields.itemmeasures \
$(CONFIGDIR)/user-naturalstories-directory.txt | genmodel
	python $(word 3, $^) <(cat $< | perl -ne 's/,(?=[A-Z]|[a-z]|[0-9])/ /g;print;' | python $(word 2, $^) WorkerId subject RT fdur) \
	$(word 4, $^) word zone item > $@
  
# Just here to preserve old buggy dev/test split
genmodel/naturalstories.fp.src.o.evmeasures: $(NATSTORDIR)/naturalstories_RTS/processed_RTs.csv \
$(RT-SCRIPTS)/rename_cols.py $(RT-SCRIPTS)/merge_tables.py genmodel/naturalstories.mfields.o.itemmeasures \
$(CONFIGDIR)/user-naturalstories-directory.txt | genmodel
	python $(word 3, $^) <(cat $< | perl -ne 's/,(?=[A-Z]|[a-z]|[0-9])/ /g;print;' | python $(word 2, $^) WorkerId subject RT fdur) \
	$(word 4, $^) word zone item > $@

# genmodel/naturalstories.gold.linetrees: genmodel/naturalstories.gcg15.decoupled.linetrees | genmodel
	# cat $< > $@
  
%naturalstories.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-NATSTORIES)/srcmodel/naturalstories.stripped.linetrees %naturalstories.linetoks
	python $^ > $@
