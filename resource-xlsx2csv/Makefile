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
CONFIG := $(CONFIGDIR)/user-xlsx2csv-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
XLSX2CSVDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(XLSX2CSVDIR))))
RESOURCE-NAME := xlsx2csv
define RESOURCE-DESCR =

NAME: xlsx2cxv
MB POINTER FILE: config/user-xlsx2csv-directory.txt
AVAILABILITY: Free
URL: https://github.com/dilshod/xlsx2csv
DESCRIPTION: XLS to CSV converter
endef
XLSX2CSV-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
XLSX2CSV-ERROR:
	$(error $(XLSX2CSV-ERROR))
DEP-XLSX2CSV-NOWILDCARD = XLSX2CSV-ERROR
endif
endif
    
%/user-xlsx2csv-directory.txt: | %   
	echo '/data/compling/xlsx2csv' > $@

################################################################################
#
#  Recipes
#
################################################################################

$(RESOURCE-XLSX2CSV)/scripts:
	mkdir $(RESOURCE-XLSX2CSV)/scripts

$(XLSX2CSV-SCRIPTS)/xlsx2csv.py: $$(DEP-XLSX2CSV-NOWILDCARD) $(XLSX2CSVDIR)/xlsx2csv.py
	mkdir -p $(XLSX2CSV-SCRIPTS)
	cp $(word 1, $^) $@

%.fromxlsx.csv: %.xlsx $(XLSX2CSV-SCRIPTS)/xlsx2csv.py
	python $(word 2, $^) -d ' ' $(word 1, $^) > $@

