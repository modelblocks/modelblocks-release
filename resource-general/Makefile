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
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

#  Resource-* directory variables.
#  Includes to these resources and pointers to files
#  contained within them should use these variables
#  rather than relative/absolute paths, so that updates
#  to variable values here will propagate to all
#  clients automatically.

MBDIR := $(realpath $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))/..)
CONFIGDIR := $(abspath $(MBDIR)/config)

RESOURCE-BNC := $(abspath $(MBDIR)/resource-bnc)
BNC-SCRIPTS := $(RESOURCE-BCN)/scripts

RESOURCE-BMMM := $(abspath $(MBDIR)/resource-bmmm)
BMMM-SCRIPTS := $(RESOURCE-BMMM)/scripts

RESOURCE-CCL := $(abspath $(MBDIR)/resource-ccl)
CCL-SCRIPTS := $(RESOURCE-CCL)/scripts

RESOURCE-CHILDES := $(abspath $(MBDIR)/resource-childes)
CHILDES-SCRIPTS := $(RESOURCE-CHILDES)/scripts

RESOURCE-DMV := $(abspath $(MBDIR)/resource-dmv)
DMV-SCRIPTS := $(RESOURCE-DMV)/scripts

RESOURCE-DUNDEE := $(abspath $(MBDIR)/resource-dundee)
DUNDEE-SCRIPTS := $(RESOURCE-DUNDEE)/scripts

RESOURCE-GCG := $(abspath $(MBDIR)/resource-gcg)
GCG-SCRIPTS := $(RESOURCE-GCG)/scripts

RESOURCE-GENERAL := $(abspath $(MBDIR)/resource-general)
GENERAL-SCRIPTS := $(RESOURCE-GENERAL)/scripts

RESOURCE-GIGAWORD := $(abspath $(MBDIR)/resource-gigaword)
GIGAWORD-SCRIPTS := $(RESOURCE-GIGAWORD)/scripts

RESOURCE-INCRSEM := $(abspath $(MBDIR)/resource-incrsem)
INCRSEM-SCRIPTS := $(RESOURCE-INCRSEM)/scripts

RESOURCE-KENLM := $(abspath $(MBDIR)/resource-kenlm)
KENLM-SCRIPTS := $(RESOURCE-KENLM)/scripts

RESOURCE-LCPARSE := $(abspath $(MBDIR)/resource-lcparse)
LCPARSE-SCRIPTS := $(RESOURCE-LCPARSE)/scripts

RESOURCE-LTREES := $(abspath $(MBDIR)/resource-linetrees)
LTREES-SCRIPTS := $(RESOURCE-LTREES)/scripts

RESOURCE-LMEFIT := $(abspath $(MBDIR)/resource-lmefit)
LMEFIT-SCRIPTS := $(RESOURCE-LMEFIT)/scripts

RESOURCE-LOGREG := $(abspath $(MBDIR)/resource-logreg)
LOGREG-SCRIPTS := $(RESOURCE-LOGREG)/scripts

RESOURCE-LVPCFG := $(abspath $(MBDIR)/resource-lvpcfg)
LVPCFG-SCRIPTS := $(RESOURCE-LVPCFG)/scripts

RESOURCE-NATSTORIES := $(abspath $(MBDIR)/resource-naturalstories)
NATSTORIES-SCRIPTS := $(RESOURCE-NATSTORIES)/scripts

RESOURCE-ONTONOTES := $(abspath $(MBDIR)/resource-ontonotes)
ONTONOTES-SCRIPTS := $(RESOURCE-ONTONOTES)/scripts

RESOURCE-RHACKS := $(abspath $(MBDIR)/resource-rhacks)
RHACKS-SCRIPTS := $(RESOURCE-RHACKS)/scripts

RESOURCE-RT := $(abspath $(MBDIR)/resource-rt)
RT-SCRIPTS := $(RESOURCE-RT)/scripts

RESOURCE-TOKENIZE := $(abspath $(MBDIR)/resource-segment-tokenize)
TOKENIZE-SCRIPTS := $(RESOURCE-TOKENIZE)/scripts

RESOURCE-SRILM := $(abspath $(MBDIR)/resource-srilm)
SRILM-SCRIPTS := $(RESOURCE-SRILM)/scripts

RESOURCE-TOKENIZER := $(abspath $(MBDIR)/resource-tokenizer)
TOKENIZER-SCRIPTS := $(RESOURCE-TOKENIZER)/scripts

RESOURCE-TREEBANK := $(abspath $(MBDIR)/resource-treebank)
TREEBANK-SCRIPTS := $(RESOURCE-TREEBANK)/scripts

RESOURCE-UCL := $(abspath $(MBDIR)/resource-ucl)
UCL-SCRIPTS := $(RESOURCE-UCL)/scripts

RESOURCE-UHHMM := $(abspath $(MBDIR)/resource-uhhmm)
UHHMM-SCRIPTS := $(RESOURCE-UHHMM)/scripts

RESOURCE-UPPARSE := $(abspath $(MBDIR)/resource-upparse)
UPPARSE-SCRIPTS := $(RESOURCE-UPPARSE)/scripts

BIN      := bin
GENMODEL := genmodel

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-cflags.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CFLAGS := $(shell cat $(CONFIG))

CONFIG := $(CONFIGDIR)/user-numthreads.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
NUMTHREADS := $(shell cat $(CONFIG))
endif

#### c++ compile flags
%/user-cflags.txt: | %
	echo '-DNDEBUG -O3' > $@

#### number of threads to use in project
%/user-numthreads.txt: | %
	echo '10' > $@

################################################################################
#
#  Printout
#
#  to construct the following file types:
#    <x>.enscr.pdf : enscripted pdf version of text file, for electronic markup
#
################################################################################

#### pdf formatted printouts for electronic markup
%.enscr.ps: %
	cat $< | enscript -fCourier7 -r -o $@
%.enscrln.ps: %
	cat $< | grep -n '' | enscript -fCourier7 -r -o $@
%.pdf: %.ps
	ps2pdf $< $@

################################################################################
#
#  Common order-only prerequisites
#
################################################################################

genmodel:
	if [ ! -d $@ ]; then mkdir $@; fi

%config:
	if [ ! -d $@ ]; then mkdir $@; fi
