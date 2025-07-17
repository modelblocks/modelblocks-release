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
.DELETE_ON_ERROR:

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-DUNDEE)

MECOW1LANGS = du ee en fi ge gr he it ko no ru sp tr
MECOW2LANGS = ba bp sp_ch da ge_po ge_zu hi_iiith hi_iitk ic no ru_mo se tr en_uk


################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-meco2-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
MECO2DIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(MECO2DIR))))
RESOURCE-NAME := The multilingual eye-movement corpus (MECO)
define RESOURCE-DESCR =

NAME: The multilingual eye-movement corpus (v1.3 of wave 1, v1.0 of wave 2)
MB POINTER FILE: config/user-meco2-directory.txt
AVAILABILITY: https://osf.io/3527a/
DESCRIPTION: A multilingual corpus of eye-tracking measures from subjects who read
12 stories (Siegelman et al., 2022).
endef
MECO2-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-MECO2 = $(error $(MECO2-ERROR))
endif
endif


%/user-meco2-directory.txt: | %
	echo '/fs/project/schuler.77/meco2' > $@


################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
define build_meco_w1_part_params
%$(1)meco_w1.part_params.ini: $(MECO2-SCRIPTS)/meco.part_params.ini
	cp $$< $$@
endef

define build_meco_w2_part_params
%$(1)meco_w2.part_params.ini: $(MECO2-SCRIPTS)/meco.part_params.ini
	cp $$< $$@
endef

define build_meco_w1_evmeasures
%$(1)meco_w1.evmeasures: $(MECO2DIR)/joint_l1_fixation_version2.0.rda $(MECO2DIR)/joint_l1_acc_breakdown.rda $(MECO2-SCRIPTS)/preprocess_meco.R $(MECO2-SCRIPTS)/process_meco.py %$(1)meco_w1.sentitems
	Rscript $$(word 3,$$^) $$(word 1,$$^) $$(word 2,$$^) $(1) | python3 $$(word 4,$$^) $$(word 5,$$^) > $$@
endef

define build_meco_w2_evmeasures
%$(1)meco_w2.evmeasures: $(MECO2DIR)/joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA $(MECO2DIR)/joint_l1_Wave2_acc_breakdown_trimmed.rda $(MECO2-SCRIPTS)/preprocess_meco.R $(MECO2-SCRIPTS)/process_meco.py %$(1)meco_w2.sentitems
	Rscript $$(word 3,$$^) $$(word 1,$$^) $$(word 2,$$^) $(1) | python3 $$(word 4,$$^) $$(word 5,$$^) > $$@
endef


################################################################################
#
#  Sents files
#
################################################################################

define build_meco_w1_sentitems
.PRECIOUS: %$(1)meco_w1.sentitems
%$(1)meco_w1.sentitems: $(MECO2DIR)/joint_l1_fixation_version2.0.rda $(MECO2-SCRIPTS)/build_meco_sentitems.R
	Rscript $$(word 2,$$^) $$(word 1,$$^) $(1) > $$@
endef

define build_meco_w2_sentitems
.PRECIOUS: %$(1)meco_w2.sentitems
%$(1)meco_w2.sentitems: $(MECO2DIR)/joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA $(MECO2-SCRIPTS)/build_meco_sentitems.R
	Rscript $$(word 2,$$^) $$(word 1,$$^) $(1) > $$@
endef

%meco_w1.itemmeasures: %meco_w1.sentitems $(MECO2-SCRIPTS)/lineitems2itemmeasures.py
	cat $<  |  python3 $(word 2, $^) | sed 's/"//g' >  $@

%meco_w2.itemmeasures: %meco_w2.sentitems $(MECO2-SCRIPTS)/lineitems2itemmeasures.py
	cat $<  |  python3 $(word 2, $^) | sed 's/"//g' >  $@

$(foreach i,$(MECOW1LANGS),\
	$(eval $(call build_meco_w1_sentitems,$(i)))\
	$(eval $(call build_meco_w1_evmeasures,$(i)))\
	$(eval $(call build_meco_w1_part_params,$(i)))\
)

$(foreach i,$(MECOW2LANGS),\
	$(eval $(call build_meco_w2_sentitems,$(i)))\
	$(eval $(call build_meco_w2_evmeasures,$(i)))\
	$(eval $(call build_meco_w2_part_params,$(i)))\
)
