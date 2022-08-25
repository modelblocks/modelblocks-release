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

define ParamVal
NaturalStoriesFMRI:
  kernel: fMRICorpus
  text: Natural Stories (fMRI corpus)
  value: naturalstoriesFMRI
endef

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-NATSTORIESFMRI)

NATSTOR-STORIES := Boar Aqua MatchstickSeller KingOfBirds Elvis MrSticky HighSchool Roswell Tulips Tourettes
NATSTOR-ROI := LPostTemp LPostTempZ LAntTemp LAntTempZ LIFG LIFGZ LIFGorb LIFGorbZ LMFG LMFGZ LAngG LAngGZ LHip RHip Hip LHipZ RHipZ HipZ
NATSTORIESFMRI-BASENAME := genmodel/naturalstoriesfmri22.wsj02to21-gcg15-nol-prtrm-3sm-synproc-+c_+u_+b5000.syn.5-kenlm.hrf
NATSTORIESFMRI-LMEDEFAULT := -Iuvw
$(eval $(call FMRI-HRF-CONTROLS,naturalstoriesfmri2,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),fmriEmpty,$(NATSTORIESFMRI-LMEDEFAULT)))
$(eval $(call FMRI-HRF-EXPERIMENTS,naturalstoriesfmri2,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),fmriEmpty,$(NATSTORIESFMRI-LMEDEFAULT)))
$(eval $(call FMRI-HRF-EXPERIMENTS,naturalstoriesfmri2,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),nsfMRI,$(NATSTORIESFMRI-LMEDEFAULT)))

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-naturalstories-fmri2-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
NATSTORFMRI2DIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(NATSTORFMRIDIR))))
RESOURCE-NAME := The Natural Stories Corpus (fMRI)
define RESOURCE-DESCR =

NAME: The Natural Stories Corpus (fMRI)
MB POINTER FILE: config/user-naturalstories-fmri-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: fMRI data collected from acoustic presentation of the
Natural Stories corpus
endef
NATSTORIESFMRI2-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-NATSTORIESFMRI2 = $(error $(NATSTORIESFMRI2-ERROR))
endif
endif



################################################################################
#
#  Recipes
#
################################################################################

## copy partition params into genmodel
genmodel/%.ini: $(NATSTORIESFMRI2-SCRIPTS)/%.ini
	cp $< $@

genmodel/%_Lang.part_params.ini: genmodel/%.part_params.ini
	cp $< $@

genmodel/%_LangRH.part_params.ini: genmodel/%.part_params.ini
	cp $< $@

genmodel/%_MD.part_params.ini: genmodel/%.part_params.ini
	cp $< $@

genmodel/%_Lang_MD.part_params.ini: genmodel/%.part_params.ini
	cp $< $@

# Link fMRI data to NatStor text preprocessing pipeline

%naturalstoriesfmri2.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri2.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri2%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri2%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri2.evmeasures: $$(DEP-NATSTORIES) $$(DEP-NATSTORIESFMRI2) $(NATSTORFMRI2DIR)/NaturalStories_TimeSeries_NpsySubs_20220823.csv $(NATSTORIESFMRI2-SCRIPTS)/preprocess_natstor_fmri.py \
$(CONFIGDIR)/user-naturalstories-fmri2-directory.txt $(NATSTORIES-SCRIPTS)/get_maxstimtime.py \
%naturalstories.t.itemmeasures | %
	cat $(word 1, $^) | python $(word 2, $^) $(NATSTOR-STORIES) | python $(word 4, $^) $(word 5, $^) > $@

%.soundpower.prdmeasures: $$(DEP-NATSTORIES) %.evmeasures $(NATSTORIESFMRI2-SCRIPTS)/get_sound_power.py $(CONFIGDIR)/user-naturalstories-directory.txt $(NATSTORDIR)/audio/*.wav
	cat $(word 1, $^) | python $(word 2, $^) $(wordlist 4, $(words $^), $^) -I 0.250 > $@

# Link fMRI data (melted and filtered to Lang LH only) to NatStor text preprocessing pipeline

%naturalstoriesfmri2_Lang.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri2_Lang.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri2_Lang%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri2_Lang%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri2_Lang.evmeasures: %naturalstoriesfmri2.evmeasures $(NATSTORIESFMRI2-SCRIPTS)/filter_network.py
	cat $(word 1, $^) | python $(word 2, $^) 1  > $@

# Link fMRI data (melted and filtered to MDlangloc only) to NatStor text preprocessing pipeline

%naturalstoriesfmri2_MD.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri2_MD.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri2_MD%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri2_MD%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri2_MD.evmeasures: %naturalstoriesfmri2.evmeasures $(NATSTORIESFMRI2-SCRIPTS)/filter_network.py
	cat $(word 1, $^) | python $(word 2, $^) 0 > $@

# Link fMRI data (melted and filtered to Lang and MDlangloc only) to NatStor text preprocessing pipeline

%naturalstoriesfmri2_Lang_MD.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri2_Lang_MD.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri2_Lang_MD%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri2_Lang_MD%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri2_Lang_MD.evmeasures: %naturalstoriesfmri2.evmeasures
	cp $^ 

%.sp.itemmeasures: $$(DEP-NATSTORIES) %.itemmeasures $(NATSTORIESFMRI-SCRIPTS)/get_sound_power.py $(CONFIGDIR)/user-naturalstories-directory.txt $(NATSTORDIR)/audio/*.wav
	cat $(word 1, $^) | python $(word 2, $^) $(wordlist 4, $(words $^), $^) > $@

