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

scripts/%.lmerform:  $(PREDRWHO-SCRIPTS)/%.lmerform  |  scripts
	cp $^ $@

# sentitems
%tree.sentitems: $(RESOURCE-PREDRWHO)/tree_sentences | genmodel
	cp $< $@

%dinner.sentitems: $(RESOURCE-PREDRWHO)/dinner_sentences | genmodel
	cp $< $@

%jeanne.sentitems: $(RESOURCE-PREDRWHO)/jeanne_sentences | genmodel
	cp $< $@

# itemmeasures
%tree.itemmeasures: %tree.sentitems $(RT-SCRIPTS)/lineitems2itemmeasures.py $(PREDRWHO-SCRIPTS)/add_docid.py | genmodel
	cat $< | python $(word 2, $^) | python $(word 3, $^) Tree > $@

%dinner.itemmeasures: %dinner.sentitems $(RT-SCRIPTS)/lineitems2itemmeasures.py $(PREDRWHO-SCRIPTS)/add_docid.py | genmodel
	cat $< | python $(word 2, $^) | python $(word 3, $^) Dinner > $@

%jeanne.itemmeasures: %jeanne.sentitems $(RT-SCRIPTS)/lineitems2itemmeasures.py $(PREDRWHO-SCRIPTS)/add_docid.py | genmodel
	cat $< | python $(word 2, $^) | python $(word 3, $^) Jeanne > $@

# senttoks
%tree.senttoks: %tree.sentitems $(TOKENIZERDIR)/ptb_tokenizer.sed | genmodel
	cat $<  |  $(word 2, $^)  |  sed "s/^' */' /;s/'Ex/' Ex/g;s/o ' clock/o'clock/g;s/gon na/gonna/g"  |  sed 's/! ARTICLE/!ARTICLE/'  |  perl -pe 's/ +$$//g'  >  $@

%dinner.senttoks: %dinner.sentitems $(TOKENIZERDIR)/ptb_tokenizer.sed | genmodel
	cat $<  |  $(word 2, $^)  |  sed "s/^' */' /;s/'Ex/' Ex/g;s/o ' clock/o'clock/g;s/gon na/gonna/g"  |  sed 's/! ARTICLE/!ARTICLE/'  |  perl -pe 's/ +$$//g'  >  $@

%jeanne.senttoks: %jeanne.sentitems $(TOKENIZERDIR)/ptb_tokenizer.sed | genmodel
	cat $<  |  $(word 2, $^)  |  sed "s/^' */' /;s/'Ex/' Ex/g;s/o ' clock/o'clock/g;s/gon na/gonna/g"  |  sed 's/! ARTICLE/!ARTICLE/'  |  perl -pe 's/ +$$//g'  >  $@

# senttrees
%tree.casp.senttrees: %tree.wsj02to21-gcg15-prtrm-4sm-fullberk-parsed.noprtrm.long.senttrees | genmodel
	cp $< $@

%dinner.casp.senttrees: %dinner.wsj02to21-gcg15-prtrm-4sm-fullberk-parsed.noprtrm.long.senttrees | genmodel
	cp $< $@

%jeanne.casp.senttrees: %jeanne.wsj02to21-gcg15-prtrm-4sm-fullberk-parsed.noprtrm.long.senttrees | genmodel
	cp $< $@

# .t.itemmeasures
%tree.t.itemmeasures: %tree.itemmeasures $(RESOURCE-PREDRWHO)/tree_word_times $(PREDRWHO-SCRIPTS)/prepare_time_itemmeasures.py | genmodel
	python $(word 3, $^) $< $(word 2, $^) > $@

%dinner.t.itemmeasures: %dinner.itemmeasures $(RESOURCE-PREDRWHO)/dinner_word_times $(PREDRWHO-SCRIPTS)/prepare_time_itemmeasures.py | genmodel
	python $(word 3, $^) $< $(word 2, $^) > $@

%jeanne.t.itemmeasures: %jeanne.itemmeasures $(RESOURCE-PREDRWHO)/jeanne_word_times $(PREDRWHO-SCRIPTS)/prepare_time_itemmeasures.py | genmodel
	python $(word 3, $^) $< $(word 2, $^) > $@

# evmeasures
%tree.evmeasures: $(NATSTORFMRIDIR)/NaturalStories_TimeSeries_NoFilter_QA_20211203.csv $(NATSTORIESFMRI-SCRIPTS)/prepare_evmeasures.py | genmodel
	cat <(head -n1 $<) <(grep ISCmega_tree $< | sed s/ISCmega_tree/Tree/) | python $(word 2, $^) > $@

%dinner.evmeasures: $(NATSTORFMRIDIR)/NaturalStories_TimeSeries_NoFilter_QA_20211203.csv $(NATSTORIESFMRI-SCRIPTS)/prepare_evmeasures.py | genmodel
	cat <(head -n1 $<) <(grep ISCmega_dinner $< | sed s/ISCmega_dinner/Dinner/) | python $(word 2, $^) > $@

%jeanne.evmeasures: $(NATSTORFMRIDIR)/NaturalStories_TimeSeries_NoFilter_QA_20211203.csv $(NATSTORIESFMRI-SCRIPTS)/prepare_evmeasures.py | genmodel
	cat <(head -n1 $<) <(grep ISCmega_jeanne  $< | sed s/ISCmega_jeanne/Jeanne/) | python $(word 2, $^) > $@


# per-region linetoks
%tree_Aud.linetoks: %tree.linetoks
	cp $(word 1, $^) $@

%tree_Lang.linetoks: %tree.linetoks
	cp $(word 1, $^) $@

%tree_MD.linetoks: %tree.linetoks
	cp $(word 1, $^) $@

%dinner_Aud.linetoks: %dinner.linetoks
	cp $(word 1, $^) $@

%dinner_Lang.linetoks: %dinner.linetoks
	cp $(word 1, $^) $@

%dinner_MD.linetoks: %dinner.linetoks
	cp $(word 1, $^) $@

%jeanne_Aud.linetoks: %jeanne.linetoks
	cp $(word 1, $^) $@

%jeanne_Lang.linetoks: %jeanne.linetoks
	cp $(word 1, $^) $@

%jeanne_MD.linetoks: %jeanne.linetoks
	cp $(word 1, $^) $@

# per-region lineitems
%tree_Aud.lineitems: %tree.lineitems
	cp $(word 1, $^) $@

%tree_Lang.lineitems: %tree.lineitems
	cp $(word 1, $^) $@

%tree_MD.lineitems: %tree.lineitems
	cp $(word 1, $^) $@

%dinner_Aud.lineitems: %dinner.lineitems
	cp $(word 1, $^) $@

%dinner_Lang.lineitems: %dinner.lineitems
	cp $(word 1, $^) $@

%dinner_MD.lineitems: %dinner.lineitems
	cp $(word 1, $^) $@

%jeanne_Aud.lineitems: %jeanne.lineitems
	cp $(word 1, $^) $@

%jeanne_Lang.lineitems: %jeanne.lineitems
	cp $(word 1, $^) $@

%jeanne_MD.lineitems: %jeanne.lineitems
	cp $(word 1, $^) $@

# per-region tokmeasures
tree_Aud%.tokmeasures: tree%.tokmeasures
	cp $(word 1, $^) $@

tree_Lang%.tokmeasures: tree%.tokmeasures
	cp $(word 1, $^) $@

tree_MD%.tokmeasures: tree%.tokmeasures
	cp $(word 1, $^) $@

dinner_Aud%.tokmeasures: dinner%.tokmeasures
	cp $(word 1, $^) $@

dinner_Lang%.tokmeasures: dinner%.tokmeasures
	cp $(word 1, $^) $@

dinner_MD%.tokmeasures: dinner%.tokmeasures
	cp $(word 1, $^) $@

jeanne_Aud%.tokmeasures: jeanne%.tokmeasures
	cp $(word 1, $^) $@

jeanne_Lang%.tokmeasures: jeanne%.tokmeasures
	cp $(word 1, $^) $@

jeanne_MD%.tokmeasures: jeanne%.tokmeasures
	cp $(word 1, $^) $@

# per-region itemmeasures
tree_Aud%.itemmeasures: tree%.itemmeasures
	cp $(word 1, $^) $@

tree_Lang%.itemmeasures: tree%.itemmeasures
	cp $(word 1, $^) $@

tree_MD%.itemmeasures: tree%.itemmeasures
	cp $(word 1, $^) $@

dinner_Aud%.itemmeasures: dinner%.itemmeasures
	cp $(word 1, $^) $@

dinner_Lang%.itemmeasures: dinner%.itemmeasures
	cp $(word 1, $^) $@

dinner_MD%.itemmeasures: dinner%.itemmeasures
	cp $(word 1, $^) $@

jeanne_Aud%.itemmeasures: jeanne%.itemmeasures
	cp $(word 1, $^) $@

jeanne_Lang%.itemmeasures: jeanne%.itemmeasures
	cp $(word 1, $^) $@

jeanne_MD%.itemmeasures: jeanne%.itemmeasures
	cp $(word 1, $^) $@

# per-region evmeasures
%tree_Aud.evmeasures: %tree.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k Aud.* | python $(word 3, $^) - > $@

%tree_Lang.evmeasures: %tree.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k Lang.* | python $(word 3, $^) - > $@

%tree_MD.evmeasures: %tree.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k MD.* | python $(word 3, $^) - > $@

%dinner_Aud.evmeasures: %dinner.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k Aud.* | python $(word 3, $^) - > $@

%dinner_Lang.evmeasures: %dinner.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k Lang.* | python $(word 3, $^) - > $@

%dinner_MD.evmeasures: %dinner.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k MD.* | python $(word 3, $^) - > $@

%jeanne_Aud.evmeasures: %jeanne.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k Aud.* | python $(word 3, $^) - > $@

%jeanne_Lang.evmeasures: %jeanne.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k Lang.* | python $(word 3, $^) - > $@

%jeanne_MD.evmeasures: %jeanne.evmeasures $(GENERAL-SCRIPTS)/melt.py $(NATSTORIESFMRI-SCRIPTS)/add_network.py
	cat $< | python $(word 2, $^) - "bold_.*" -p 5 -k MD.* | python $(word 3, $^) - > $@

# partition parameters
genmodel/tree.part_params.ini: $(PREDRWHO-SCRIPTS)/predrwho.part_params.ini
	cp $< $@

genmodel/dinner.part_params.ini: $(PREDRWHO-SCRIPTS)/predrwho.part_params.ini
	cp $< $@

genmodel/jeanne.part_params.ini: $(PREDRWHO-SCRIPTS)/predrwho.part_params.ini
	cp $< $@

genmodel/%_Aud.part_params.ini: genmodel/%.part_params.ini
	cp $< $@

genmodel/%_Lang.part_params.ini: genmodel/%.part_params.ini
	cp $< $@

genmodel/%_MD.part_params.ini: genmodel/%.part_params.ini
	cp $< $@

# Soundpower -- these make items require you to generate
# %.soundpower.prdmeasures elsewhere
%_Aud.soundpower.prdmeasures: %.soundpower.prdmeasures %_Aud.evmeasures $(FMRI-SCRIPTS)/tile_stimuli.py 
	python $(word 3, $^) $(word 1, $^) $(word 2, $^) > $@

%_Lang.soundpower.prdmeasures: %.soundpower.prdmeasures %_Lang.evmeasures $(FMRI-SCRIPTS)/tile_stimuli.py 
	python $(word 3, $^) $(word 1, $^) $(word 2, $^) > $@

%_MD.soundpower.prdmeasures: %.soundpower.prdmeasures %_MD.evmeasures $(FMRI-SCRIPTS)/tile_stimuli.py 
	python $(word 3, $^) $(word 1, $^) $(word 2, $^) > $@


## Old soundpower stuff -- shouldn't be needed anymore
#%dinner_Aud.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/dinner/dinner.wav %dinner_Aud.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%dinner_Lang.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/dinner/dinner.wav %dinner_Lang.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%dinner_MD.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/dinner/dinner.wav %dinner_MD.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%jeanne_Aud.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/jeanne/jeanne.wav %jeanne_Aud.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%jeanne_Lang.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/jeanne/jeanne.wav %jeanne_Lang.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%jeanne_MD.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/jeanne/jeanne.wav %jeanne_MD.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%tree_Aud.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/tree/tree.wav %tree_Aud.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%tree_Lang.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/tree/tree.wav %tree_Lang.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@
#
#%tree_MD.soundpower.prdmeasures: /home/clark.3664/projects/drwho_stepping_stone/tree/tree.wav %tree_MD.evmeasures $(PREDRWHO-SCRIPTS)/get_sound_power.py
#	cat $(word 2, $^) | python $(word 3, $^) $(word 1, $^) 0.1 > $@


# shouldn't be needed any more -- this is covered by resource-fmri/scripts/tile_stimuli.py
# $(PREDRWHO-SCRIPTS)/tile_stimuli.py
#.PRECIOUS: %.fmri-bywrd.prdmeasures
#genmodel/tree%.fmri-bywrd.prdmeasures: $(PREDRWHO-SCRIPTS)/tile_stimuli.py genmodel/tree%.all-itemmeasures \
#genmodel/tree$$(word 1,$$(subst ., ,%)).evmeasures
#	python $(word 1, $^) $(word 2, $^) $(word 3, $^) > $@
#
#genmodel/dinner%.fmri-bywrd.prdmeasures: $(PREDRWHO-SCRIPTS)/tile_stimuli.py genmodel/dinner%.all-itemmeasures \
#genmodel/dinner$$(word 1,$$(subst ., ,%)).evmeasures
#	python $(word 1, $^) $(word 2, $^) $(word 3, $^) > $@
#
#genmodel/jeanne%.fmri-bywrd.prdmeasures: $(PREDRWHO-SCRIPTS)/tile_stimuli.py genmodel/jeanne%.all-itemmeasures \
#genmodel/jeanne$$(word 1,$$(subst ., ,%)).evmeasures
#	python $(word 1, $^) $(word 2, $^) $(word 3, $^) > $@
