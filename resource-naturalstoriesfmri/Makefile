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
NATSTORIESFMRI-BASENAME := genmodel/naturalstoriesfmri.wsj02to21-gcg15-nol-prtrm-3sm-synproc-+c_+u_+b5000.syn.5-kenlm.hrf
NATSTORIESFMRI-LMEDEFAULT := -Iuvw
$(eval $(call FMRI-HRF-CONTROLS,naturalstoriesfmri,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),fmriEmpty,$(NATSTORIESFMRI-LMEDEFAULT)))
$(eval $(call FMRI-HRF-EXPERIMENTS,naturalstoriesfmri,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),fmriEmpty,$(NATSTORIESFMRI-LMEDEFAULT)))
$(eval $(call FMRI-HRF-EXPERIMENTS,naturalstoriesfmri,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),nsfMRI,$(NATSTORIESFMRI-LMEDEFAULT)))

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-naturalstories-fmri-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
NATSTORFMRIDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(NATSTORFMRIDIR))))
RESOURCE-NAME := The Natural Stories Corpus (fMRI)
define RESOURCE-DESCR =

NAME: The Natural Stories Corpus (fMRI)
MB POINTER FILE: config/user-naturalstories-fmri-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: fMRI data collected from acoustic presentation of the
Natural Stories corpus
endef
NATSTORIESFMRI-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-NATSTORIESFMRI = $(error $(NATSTORIESFMRI-ERROR))
endif
endif



%/user-naturalstories-fmri-directory.txt: | %   
	echo '/data/corpora/original/english/naturalstories_fmri' > $@

################################################################################
#
#  Recipes
#
################################################################################

## copy partition params into genmodel
genmodel/%.ini: $(NATSTORIESFMRI-SCRIPTS)/%.ini
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

%naturalstoriesfmri.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

#%naturalstoriesfmri.evmeasures: $$(DEP-NATSTORIES) $$(DEP-NATSTORIESFMRI) $(NATSTORFMRIDIR)/naturalstoriesFMRI.evmeasures $(NATSTORIESFMRI-SCRIPTS)/preprocess_natstor_fmri.py \
#$(NATSTORIESFMRI-SCRIPTS)/filter_docs.py $(CONFIGDIR)/user-naturalstories-fmri-directory.txt $(NATSTORIES-SCRIPTS)/get_maxstimtime.py \
#%naturalstories.t.itemmeasures | %
#	cat $(word 1, $^) | python $(word 2, $^) $(NATSTOR-STORIES) sampleid | python $(word 3, $^) | python $(word 5, $^) $(word 6, $^) > $@

#%naturalstoriesfmri.evmeasures: /home/clark.3664/projects/drwho_stepping_stone/NaturalStories_TimeSeries_NoFilter_QA_20211203.csv $(NATSTORIESFMRI-SCRIPTS)/filter_docs.py $(NATSTORIESFMRI-SCRIPTS)/rename_rois.py $(NATSTORIESFMRI-SCRIPTS)/prepare_evmeasures.py | genmodel
%naturalstoriesfmri.evmeasures: $(NATSTORFMRIDIR)/NaturalStories_TimeSeries_NoFilter_QA_20211203.csv $(NATSTORIESFMRI-SCRIPTS)/filter_docs.py $(NATSTORIESFMRI-SCRIPTS)/prepare_evmeasures.py | genmodel
	cat $(word 1, $^) | python $(word 2, $^) | python $(word 3, $^) > $@

%naturalstoriesfmri_Lang.soundpower.prdmeasures: $$(DEP-NATSTORIES) %naturalstoriesfmri_Lang.evmeasures $(NATSTORIESFMRI-SCRIPTS)/get_sound_power.py $(CONFIGDIR)/user-naturalstories-directory.txt $(NATSTORDIR)/audio/*.wav
	cat $(word 1, $^) | python $(word 2, $^) $(wordlist 4, $(words $^), $^) -I 0.1 > $@

%naturalstoriesfmri_MD.soundpower.prdmeasures: $$(DEP-NATSTORIES) %naturalstoriesfmri_MD.evmeasures $(NATSTORIESFMRI-SCRIPTS)/get_sound_power.py $(CONFIGDIR)/user-naturalstories-directory.txt $(NATSTORDIR)/audio/*.wav
	cat $(word 1, $^) | python $(word 2, $^) $(wordlist 4, $(words $^), $^) -I 0.1 > $@

# Link fMRI data (melted and filtered to Lang LH only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_Lang.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_Lang.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_Lang%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_Lang%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_Lang.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	#cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k LangL.* | python $(word 3, $^) - > $@
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k Lang.* | python $(word 3, $^) - > $@

# Link fMRI data (melted and filtered to Lang RH only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_LangRH.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_LangRH.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_LangRH%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_LangRH%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_LangRH.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k LangR.* | python $(word 3, $^) - > $@

# Link fMRI data (melted and filtered to MDlangloc only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_MD.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_MD.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_MD%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_MD%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_MD.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	#cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k MDlangloc.* | python $(word 3, $^) - > $@
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k MDlangloc.* | python $(word 3, $^) - > $@

# Link fMRI data (melted and filtered to Lang RH only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_LangRH.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_LangRH.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_LangRH%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_LangRH%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_LangRH.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k LangR.* | python $(word 3, $^) - > $@

# Link fMRI data (melted and filtered to MDlangloc only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_MD.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_MD.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_MD%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_MD%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

# Link fMRI data (melted and filtered to MDspatWM only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_MDspatWM.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_MDspatWM.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_MDspatWM%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_MDspatWM%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_MDspatWM.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k MDspatwm.* | python $(word 3, $^) - > $@

# Link fMRI data (melted and filtered to AC only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_AC.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_AC.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_AC%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_AC%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_AC.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k AC.* | python $(word 3, $^) - > $@


# Link fMRI data (melted and filtered to Lang and MDlangloc only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_Lang_MD.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_Lang_MD.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_Lang_MD%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_Lang_MD%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_Lang_MD.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k MDlangloc.* LangL.* | python $(word 3, $^) - > $@

# Link fMRI data (melted and filtered to Lang and MDspatWM only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_Lang_MDspatWM.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_Lang_MDspatWM.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_Lang_MDspatWM%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_Lang_MDspatWM%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmri_Lang_MDspatWM.evmeasures: %naturalstoriesfmri.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k MDspatwm.* LangL.* | python $(word 3, $^) - > $@



%.sp.itemmeasures: $$(DEP-NATSTORIES) %.itemmeasures $(NATSTORIESFMRI-SCRIPTS)/get_sound_power.py $(CONFIGDIR)/user-naturalstories-directory.txt $(NATSTORDIR)/audio/*.wav
	cat $(word 1, $^) | python $(word 2, $^) $(wordlist 4, $(words $^), $^) > $@


# Link fMRI data (melted and filtered to bilateral hippocampus only) to NatStor text preprocessing pipeline

%naturalstoriesfmri_Hip.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri_Hip.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri_Hip%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri_Hip%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@

%naturalstoriesfmriHipWide.evmeasures: $$(DEP-NATSTORIES) $$(DEP-NATSTORIESFMRI) $(NATSTORFMRIDIR)/naturalstories_fmriHIP.evmeasures $(NATSTORIESFMRI-SCRIPTS)/preprocess_natstor_fmri.py \
$(NATSTORIESFMRI-SCRIPTS)/filter_docs.py $(CONFIGDIR)/user-naturalstories-fmri-directory.txt $(NATSTORIES-SCRIPTS)/get_maxstimtime.py \
%naturalstories.t.itemmeasures | %
	cat $(word 1, $^) | python $(word 2, $^) $(NATSTOR-STORIES) sampleid | python $(word 3, $^) | python $(word 5, $^) $(word 6, $^) > $@

%naturalstoriesfmri_Hip.evmeasures: %naturalstoriesfmriHipWide.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $(word 1, $^) | python $(word 2, $^) - "bold.*" -p 4 -k "[LR]Hip.*" | python $(word 3, $^) - > $@
