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

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-pereira-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
PEREIRADIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(PEREIRADIR))))
RESOURCE-NAME := Pythia Pile
define RESOURCE-DESCR =

NAME: Pereira et al. 2018
MB POINTER FILE: config/user-pereira-directory.txt
AVAILABILITY: Free
URL: https://osf.io/crwz7/
DESCRIPTION: Training data from Pereira et al. 2023
endef
PEREIRA-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-PEREIRA = $(error $(PEREIRA-ERROR))
else
DEP-PEREIRA :=
endif
endif

%/user-pereira-directory.txt: | %
	echo '/fs/project/schuler.77/corpora/original/english/pereira2018' > $@

################################################################################

## non-standard
genmodel/%.csv:  $(PEREIRA-SCRIPTS)/transform_mat.py  $(PEREIRADIR)/$$(basename %)/data_$$(subst .,,$$(suffix $$*))sentences.mat
	python3  $<  -I $(word 2,$^)  -O $@

#genmodel/pereira%.sent.resmeasures:  $(PEREIRA-SCRIPTS)/transform_mat.py  $(PEREIRADIR)/$$(word 1,$$(subst -, ,$$*))/data_$$(word 2,$$(subst -, ,$$*))sentences.mat
#	python3  $<  -I $(word 2,$^)  -O $@
genmodel/pereira%.sentitems genmodel/pereira-sent%.resmeasures:  $(PEREIRA-SCRIPTS)/get_sentences_and_voxels.py  $(PEREIRADIR)/$$(word 1,$$(subst -, ,$$*))/data_$$(word 2,$$(subst -, ,$$*))sentences.mat
	python3  $<  -I $(word 2,$^)  -S genmodel/pereira$*.sentitems  -C genmodel/pereira-sent$*.resmeasures

genmodel/pereira-sent%.prdmeasures:  $(PEREIRA-SCRIPTS)/integrate.py  genmodel/pereira-sent$$(basename %).resmeasures  genmodel/pereira%.itemmeasures
	python3  $<  -S $(word 2,$^)  -C $(word 3,$^)  -O $@


