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
CONFIG := $(CONFIGDIR)/user-pythiapile-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
PYTHIADIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(PYTHIADIR))))
RESOURCE-NAME := Pythia Pile
define RESOURCE-DESCR =

NAME: Pythia Pile
MB POINTER FILE: config/user-pythiapile-directory.txt
AVAILABILITY: Free
URL: https://github.com/EleutherAI/pythia?tab=readme-ov-file#reproducing-training
DESCRIPTION: Training data from the Pile, processed in the exact order that was used to train Pythia LMs (Biderman et al. 2023)
endef
PYTHIAPILE-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-PYTHIAPILE = $(error $(PYTHIAPILE-ERROR))
else
DEP-PYTHIAPILE :=
endif
endif

%/user-pythiapile-directory.txt: | %
	echo '/fs/project/schuler.77/corpora/pythia_pile' > $@

################################################################################
