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
#  include $(RESOURCE-SKIPDEP)/Makefile
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
CONFIG := $(CONFIGDIR)/user-skipdep-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
SKIPDEPDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(SKIPDEPDIR))))
RESOURCE-NAME := R-Hacks
define RESOURCE-DESCR =

NAME: Skip-dep embeddings from Bansal et al. (2014)
MB POINTER FILE: config/user-skipdep-directory.txt
AVAILABILITY: Free
URL: http://ttic.uchicago.edu/~mbansal/codedata/dependencyEmbeddings-skipdep.zip
DESCRIPTION: Skip-dep embeddings from Bansal et al. (2014)
endef
SKIPDEP-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-SKIPDEP = $(error $(SKIPDEP-ERROR))
else
DEP-SKIPDEP :=
endif
endif
    
%/user-skipdep-directory.txt: | %   
	echo '/project/lin-dept/compling/skipdep' > $@

################################################################################
#
#  Recipes
#
################################################################################

