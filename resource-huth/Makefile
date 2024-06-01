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
#  include $(RESOURCE-HUTH)/Makefile
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
CONFIG := $(CONFIGDIR)/user-huth-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
HUTHDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(HUTHDIR))))
RESOURCE-NAME := Huth Speech Model Tutorial
define RESOURCE-DESCR =

NAME: Huth Speech Model Tutorial
MB POINTER FILE: config/user-huth-directory.txt
AVAILABILITY: Free
URL: https://github.com/HuthLab/speechmodeltutorial
DESCRIPTION: fMRI speech processing code from Alex Huth's EACL14 tutorial
endef
HUTH-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-HUTH = $(error $(HUTH-ERROR))
else
DEP-HUTH :=
endif
endif
    
%/user-huth-directory.txt: | %   
	echo '/project/lin-dept/compling/speechmodeltutorial' > $@

################################################################################
#
#  Recipes
#
################################################################################

$(RESOURCE-HUTH)/scripts:
	mkdir $(RESOURCE-HUTH)/scripts

.PRECIOUS: $(HUTH-SCRIPTS)/%.py
$(HUTH-SCRIPTS)/%.py: $$(DEP-HUTH) $(HUTHDIR)/%.py | $(HUTH-SCRIPTS)
	cp $(word 1, $^) $@


