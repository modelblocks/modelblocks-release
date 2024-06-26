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
#  include $(RESOURCE-RHACKS)/Makefile
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

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-rhacks-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
RHACKSDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(RHACKSDIR))))
RESOURCE-NAME := R-Hacks
define RESOURCE-DESCR =

NAME: R-Hacks
MB POINTER FILE: config/user-rhacks-directory.txt
AVAILABILITY: Free
URL: https://github.com/aufrank/R-hacks
DESCRIPTION: Useful bits of code for programming and analysis in R.
endef
RHACKS-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-RHACKS = $(error $(RHACKS-ERROR))
else
DEP-RHACKS :=
endif
endif
    
%/user-rhacks-directory.txt: | %   
	echo '/fs/project/schuler.77/compling/R-hacks' > $@

################################################################################
#
#  Recipes
#
################################################################################

$(RESOURCE-RHACKS)/scripts:
	mkdir $(RESOURCE-RHACKS)/scripts

.PRECIOUS: $(RHACKS-SCRIPTS)/%.R
$(RHACKS-SCRIPTS)/%.R: $$(DEP-RHACKS) $(RHACKSDIR)/%.R | $(RHACKS-SCRIPTS)
	cp $(word 1, $^) $@

