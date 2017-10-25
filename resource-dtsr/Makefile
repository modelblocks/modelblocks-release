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
#  Include statements to this file should be preceded by includes to any 
#  Makefiles needed to produce the data to evaluate: <data-name>.full.evmeasures. 
#  For reading time experiments, this is usually resource-rt/Makefile and all 
#  Makefiles used to generate and combine experiment-specific data.
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

VPATH += $(RESOURCE-DTSR)

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-dtsr-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
DTSRDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(DTSRDIR))))
RESOURCE-NAME := Deconvolutional Time Series Regression (DTSR)
define RESOURCE-DESCR =

NAME: Deconvolutional Time Series Regression (DTSR)
MB POINTER FILE: config/user-dtsr-directory.txt
AVAILABILITY: Public
DESCRIPTION: A toolkit for jointly learning effect estimates and
temporal convolution shapes from time series data. 
endef
DTSR-ERROR := $(call DEPWARN, $(DTSR-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-DTSR = $(error $(DTSR-ERROR))
endif
endif


%/user-dtsr-directory.txt: | %   
	echo '/home/compling/dtsr' > $@


################################################################################
#
#  Recipes 
#
################################################################################

.PRECIOUS: %.t.core.evmeasures
%.t.core.evmeasures: %.core.evmeasures $$(DTSR-SCRIPTS)/compute_timestamps_from_RT.py
	cat $(word 1, $^) | python $(word 2, $^) > $@

%.y.evmeasures: %.evmeasures $$(RT-SCRIPTS)/filter_cols.awk
	cat $(word 1, $^) | awk -f $(word 2, $^) -v cols=fdur:fdurFP:fdurGP:time:subject:docid:sentid:word:correct:startofsentence:endofsentence:startofline:endofline:startofscreen:endofscreen:startoffile:endoffile:wdelta > $@

