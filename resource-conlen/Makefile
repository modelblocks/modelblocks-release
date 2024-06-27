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
	echo '/fs/project/lin-dept/corpora/original/english/constituent_length' > $@

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

conlen%.itemmeasures: $$(DEP-CONLEN) $(CONLENDIR)/conlen$$*_src.itemmeasures
	cp $< $@

conlen%fmri.beh.evmeasures: $$(DEP-CONLEN) $(CONLENDIR)/conlen$$*fmri_beh.csv $(CONFIGDIR)/user-conlen-directory.txt $(CONLEN-SCRIPTS)/build_itemmeasures.py | %
	cat $< | python $(word 3, $^) > $@

conlen%fmri.wide.hrf.evmeasures: $(FMRI-SCRIPTS)/hrf_convolve_predictors.py conlen%fmri.beh.evmeasures
	python $^ -g subject run > $@

conlen%c.itemmeasures: $(CONLEN-SCRIPTS)/filter_itemmeasures.py conlen%.itemmeasures
	python $^ c > $@

conlen%nc.itemmeasures: $(CONLEN-SCRIPTS)/filter_itemmeasures.py conlen%.itemmeasures
	python $^ nc > $@

conlen%jab.itemmeasures: $(CONLEN-SCRIPTS)/filter_itemmeasures.py conlen%.itemmeasures
	python $^ jab > $@

conlen%c.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/conlen$$*_c.lineitems
	cp $< $@

conlen%nc.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/conlen$$*_nc.lineitems
	cp $< $@

conlen%ncsents.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/conlen$$*_ncsents.lineitems
	cp $< $@

conlen%jab.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/conlen$$*_jab.lineitems
	cp $< $@

conlen%c.linetoks: $$(DEP-TOKENIZER) conlen%c.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

conlen%nc.linetoks: $$(DEP-TOKENIZER) conlen%nc.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

conlen%ncsents.linetoks: $$(DEP-TOKENIZER) conlen%ncsents.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

conlen%jab.linetoks: $$(DEP-TOKENIZER) conlen%jab.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

conlen%c.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-CONLEN)/srcmodel/conlen$$*c.stripped.linetrees conlen%c.linetoks
	python $^ > $@

%conlen2c.12.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlen2c.stripped.linetrees
	head -n 30 $^ > $@

%conlen2c.6.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlen2c.stripped.linetrees
	head -n 90 $^ | tail -n 60 > $@

%conlen2c.4.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlen2c.stripped.linetrees
	head -n 180 $^ | tail -n 90 > $@

%conlen2c.3.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlen2c.stripped.linetrees
	head -n 300 $^ | tail -n 120 > $@

%conlen2c.2.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlen2c.stripped.linetrees
	head -n 480 $^ | tail -n 180 > $@

%conlen2c.1.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlen2c.stripped.linetrees
	tail -n 360 $^ > $@

%conlen2jab.stripped.linetrees: %conlen2c.12.stripped.linetrees %conlen2c.4.stripped.linetrees %conlen2c.1.stripped.linetrees
	cat $^ > $@

%conlen2jab.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py %conlen2jab.stripped.linetrees %conlen2jab.linetoks
	python $^ > $@

%conlen2ncsents.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-CONLEN)/srcmodel/conlen2ncsents.stripped.linetrees %conlen2ncsents.linetoks
	python $^ > $@

conlen2ncfilt%itemmeasures: $$(dir %)/conlen2nc.itemmeasures conlen2ncsents%itemmeasures $(CONLEN-SCRIPTS)/build_nc.py $$(dir %)/conlen2nc.lineitems \
	$(RT-SCRIPTS)/evmeas2uniqcols.py
	paste -d ' ' $< <(cat $(word 2, $^) | python $(word 3, $^) $(word 4, $^)) | python $(word 5, $^) > $@

conlen2fmri%itemmeasures: $(CONLEN-SCRIPTS)/concat_itemmeasures.py conlen2c%itemmeasures conlen2ncfilt%itemmeasures conlen2jab%itemmeasures
	python $^ > $@

# %conlenfmri_bold.evmeasures: $$(DEP-CONLEN) $(CONLENDIR)/conlenfmri_bold.csv $(CONFIGDIR)/user-conlen-directory.txt | %
# 	cp $< $@
# 
# %conlenfmri.resmeasures: $$(DEP-CONLEN) %conlenfmri_bold.evmeasures $(CONLEN-SCRIPTS)/preprocess_conlen_fmri.py
# 	cat $(word 1, $^) | python $(word 2, $^) > $@
# 
# %conlenfmri_LANG.resmeasures: %conlenfmri.resmeasures $(GENERAL-SCRIPTS)/melt.py $(CONLEN-SCRIPTS)/add_network.py
# 	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k LANG.* | python $(word 3, $^) > $@
# 
# %conlenfmri_MD.resmeasures: %conlenfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(CONLEN-SCRIPTS)/add_network.py
# 	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k MD.* | python $(word 3, $^)  > $@
# 
# %conlenfmri_LANG_MD.resmeasures: %conlenfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(CONLEN-SCRIPTS)/add_network.py
# 	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k LANG.* MD.* | python $(word 3, $^)  > $@
# 
# %conlenfmri.hrf.prdmeasures: $$(DEP-CONLEN) $(CONLENDIR)/conlenfmri_beh.csv
# 	cp $^ $@
# 
# %conlenfmri_LANG.hrf.prdmeasures: $(CONLEN-SCRIPTS)/tile_stim_into_roi.py %conlenfmri_LANG.evmeasures %conlenfmri.wide.hrf.evmeasures
# 	python $^ > $@
# 
# %conlenfmri_MD.hrf.prdmeasures: $(CONLEN-SCRIPTS)/tile_stim_into_roi.py %conlenfmri_MD.evmeasures %.conlenfmri.wide.hrf.evmeasures
# 	python $^ > $@
# 
# %conlenfmri_LANG_MD.hrf.prdmeasures: $(CONLEN-SCRIPTS)/tile_stim_into_roi.py %conlenfmri_LANG_MD.evmeasures %conlenfmri.wide.hrf.evmeasures
# 	python $^ > $@
# 
# %hrf.all.evmeasures: $(CONLEN-SCRIPTS)/merge_prd_res.py %hrf.prdmeasures %resmeasures
# 	python $^ > $@
# 
