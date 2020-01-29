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
#  include $(RESOURCE-NATSTORIES)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-conlen-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CONLENDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(CONLENDIR))))
RESOURCE-NAME := The constituent length fMRI corpus
define RESOURCE-DESCR =

NAME: The constituent length fMRI corpus
MB POINTER FILE: config/user-conlen-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: Materials used to study effects of constituent length on brain responses
endef
CONLEN-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-CONLEN = $(error $(CONLEN-ERROR))
endif
endif



%/user-conlen-directory.txt: | %   
	echo '/data/corpora/original/english/constituent_length' > $@

################################################################################
#
#  Recipes
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(CONLEN-SCRIPTS)/%.ini
	cp $< $@

genmodel/%_LANG.ini: genmodel/%.ini
	cp $< $@

genmodel/%_LANG_MD.ini: genmodel/%.ini
	cp $< $@

genmodel/%_LANG_MD.ini: genmodel/%.ini
	cp $< $@

%conlen.itemmeasures: $$(DEP-CONLEN) $(CONLENDIR)/constituent_length.itemmeasures $(CONLEN-SCRIPTS)/build_itemmeasures.py $(CONFIGDIR)/user-conlen-directory.txt | %
	cat $< | python $(word 2, $^) > $@

%conlenc.itemmeasures: $(CONLEN-SCRIPTS)/filter_itemmeasures.py %conlen.itemmeasures
	python $^ c > $@

%conlennc.itemmeasures: $(CONLEN-SCRIPTS)/filter_itemmeasures.py %conlen.itemmeasures
	python $^ nc > $@

%conlenjab.itemmeasures: $(CONLEN-SCRIPTS)/filter_itemmeasures.py %conlen.itemmeasures
	python $^ jab > $@

%conlenc.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/constituent_length_c.lineitems
	cp $< $@

%conlennc.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/constituent_length_nc.lineitems
	cp $< $@

%conlenncsents.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/constituent_length_ncsents.lineitems
	cp $< $@

%conlenjab.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/constituent_length_jab.lineitems
	cp $< $@

%conlenc.linetoks: $$(DEP-TOKENIZER) %conlenc.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

%conlennc.linetoks: $$(DEP-TOKENIZER) %conlennc.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

%conlenncsents.linetoks: $$(DEP-TOKENIZER) %conlenncsents.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

%conlenjab.linetoks: $$(DEP-TOKENIZER) %conlenjab.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

%conlenc.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees %conlenc.linetoks
	python $^ > $@

%conlenc.12.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 30 $^ > $@

%conlenc.6.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 90 $^ | tail -n 60 > $@

%conlenc.4.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 180 $^ | tail -n 90 > $@

%conlenc.3.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 300 $^ | tail -n 120 > $@

%conlenc.2.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 480 $^ | tail -n 180 > $@

%conlenc.1.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	tail -n 360 $^ > $@

%conlenjab.stripped.linetrees: %conlenc.12.stripped.linetrees %conlenc.4.stripped.linetrees %conlenc.1.stripped.linetrees
	cat $^ > $@

%conlenjab.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py %conlenjab.stripped.linetrees %conlenjab.linetoks
	python $^ > $@

%conlenncsents.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-CONLEN)/srcmodel/conlenncsents.stripped.linetrees %conlenncsents.linetoks
	python $^ > $@

conlenncfilt%itemmeasures: $$(dir %)/conlennc.itemmeasures conlenncsents%itemmeasures $(CONLEN-SCRIPTS)/build_nc.py $$(dir %)/conlennc.lineitems \
	$(RT-SCRIPTS)/evmeas2uniqcols.py
	paste -d ' ' $< <(cat $(word 2, $^) | python $(word 3, $^) $(word 4, $^)) | python $(word 5, $^) > $@

conlenfmri%itemmeasures: $(CONLEN-SCRIPTS)/concat_itemmeasures.py conlenc%itemmeasures conlenncfilt%itemmeasures conlenjab%itemmeasures
	python $^ > $@

%conlenfmri.evmeasures: $$(DEP-CONLEN) $(CONLENDIR)/conlenfmri_src.evmeasures $(CONLEN-SCRIPTS)/preprocess_conlen_fmri.py $(CONFIGDIR)/user-conlen-directory.txt | %
	cat $(word 1, $^) | python $(word 2, $^) > $@

%conlenfmri_LANG.evmeasures: %conlenfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(CONLEN-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k LANG.* | python $(word 3, $^) > $@

%conlenfmri_MD.evmeasures: %conlenfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(CONLEN-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k MD.* | python $(word 3, $^)  > $@

%conlenfmri_LANG_MD.evmeasures: %conlenfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(CONLEN-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k LANG.* MD.* | python $(word 3, $^)  > $@

%conlenfmri.beh.evmeasures: $$(DEP-CONLEN) $(CONLENDIR)/conlenfmri_beh.txt
	cp $^ $@

conlenfmri.%.evmeasures: $(CONLEN-SCRIPTS)/merge_itemmeasures_into_runs.py $$(dir %)conlenfmri.beh.evmeasures conlenfmri.%.all-itemmeasures
	python $^ > $@

conlenfmri_LANG.%.evmeasures: $(CONLEN-SCRIPTS)/tile_stim_into_roi.py $$(dir %)conlenfmri_LANG.evmeasures conlenfmri.%.evmeasures
	python $^ > $@

conlenfmri_MD.%.evmeasures: $(CONLEN-SCRIPTS)/tile_stim_into_roi.py $$(dir %)conlenfmri_MD.evmeasures conlenfmri.%.evmeasures
	python $^ > $@

conlenfmri_LANG_MD.%.evmeasures: $(CONLEN-SCRIPTS)/tile_stim_into_roi.py $$(dir %)conlenfmri_LANG_MD.evmeasures conlenfmri.%.evmeasures
	python $^ > $@

