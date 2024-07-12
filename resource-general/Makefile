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

define CONFIGWARN
------------------------------------------------------------------
ATTENTION! I had to create config file:

$(1)

which may contain an incorrect default value.
Targets with dependencies to this file may fail until it is fixed.
------------------------------------------------------------------
endef

define DEPWARN

ERROR: You have tried to make a target that depends on the following
missing external resource:

$(1)

If you do not have this resource, you will first need to obtain it.

If the resource exists on your system and ModelBlocks cannot find
it, you may need to correct the following pointer file:

$(2)

Details about the resource are below:
$(3) 

endef

RESOURCE-ALICE := $(abspath $(MBDIR)/resource-alice)
ALICE-SCRIPTS := $(RESOURCE-ALICE)/scripts

RESOURCE-BNC := $(abspath $(MBDIR)/resource-bnc)
BNC-SCRIPTS := $(RESOURCE-BCN)/scripts

RESOURCE-BMMM := $(abspath $(MBDIR)/resource-bmmm)
BMMM-SCRIPTS := $(RESOURCE-BMMM)/scripts

RESOURCE-CCL := $(abspath $(MBDIR)/resource-ccl)
CCL-SCRIPTS := $(RESOURCE-CCL)/scripts

RESOURCE-CHILDES := $(abspath $(MBDIR)/resource-childes)
CHILDES-SCRIPTS := $(RESOURCE-CHILDES)/scripts

RESOURCE-CONLEN := $(abspath $(MBDIR)/resource-conlen)
CONLEN-SCRIPTS := $(RESOURCE-CONLEN)/scripts

RESOURCE-COREF-CONLL := $(abspath $(MBDIR)/resource-coref-conll)
CONLL-SCRIPTS := $(RESOURCE-COREF-CONLL)/scripts

RESOURCE-DMV := $(abspath $(MBDIR)/resource-dmv)
DMV-SCRIPTS := $(RESOURCE-DMV)/scripts

RESOURCE-CDR := $(abspath $(MBDIR)/resource-cdr)
CDR-SCRIPTS := $(RESOURCE-CDR)/scripts

RESOURCE-DUNDEE := $(abspath $(MBDIR)/resource-dundee)
DUNDEE-SCRIPTS := $(RESOURCE-DUNDEE)/scripts

RESOURCE-PROVO := $(abspath $(MBDIR)/resource-provo)
PROVO-SCRIPTS := $(RESOURCE-PROVO)/scripts

RESOURCE-GECO := $(abspath $(MBDIR)/resource-geco)
GECO-SCRIPTS := $(RESOURCE-GECO)/scripts

RESOURCE-REPETITION := $(abspath $(MBDIR)/resource-repetition)
REPETITION-SCRIPTS := $(RESOURCE-REPETITION)/scripts

RESOURCE-EPIC := $(abspath $(MBDIR)/resource-epic)
EPIC-SCRIPTS := $(RESOURCE-EPIC)/scripts

RESOURCE-FEDECOG := $(abspath $(MBDIR)/resource-fedorenkoetal16ecog)
FEDECOG-SCRIPTS := $(RESOURCE-FEDECOG)/scripts

RESOURCE-FMRI := $(abspath $(MBDIR)/resource-fmri)
FMRI-SCRIPTS := $(RESOURCE-FMRI)/scripts

RESOURCE-FORREST := $(abspath $(MBDIR)/resource-forrest)
FORREST-SCRIPTS := $(RESOURCE-FORREST)/scripts

RESOURCE-GCG := $(abspath $(MBDIR)/resource-gcg)
GCG-SCRIPTS := $(RESOURCE-GCG)/scripts

RESOURCE-GENERAL := $(abspath $(MBDIR)/resource-general)
GENERAL-SCRIPTS := $(RESOURCE-GENERAL)/scripts

RESOURCE-GIGAWORD := $(abspath $(MBDIR)/resource-gigaword)
GIGAWORD-SCRIPTS := $(RESOURCE-GIGAWORD)/scripts

RESOURCE-OPENWEBTEXT := $(abspath $(MBDIR)/resource-openwebtext)
OPENWEBTEXT-SCRIPTS := $(RESOURCE-OPENWEBTEXT)/scripts

RESOURCE-GLOVE := $(abspath $(MBDIR)/resource-glove)
GLOVE-SCRIPTS := $(RESOURCE-GLOVE)/scripts

RESOURCE-GLSTM := $(abspath $(MBDIR)/resource-glstm)
GLSTM-SCRIPTS := $(RESOURCE-GLSTM)/scripts

RESOURCE-GPT2 := $(abspath $(MBDIR)/resource-gpt2)
GPT2-SCRIPTS := $(RESOURCE-GPT2)/scripts

RESOURCE-HF := $(abspath $(MBDIR)/resource-hf)
HF-SCRIPTS := $(RESOURCE-HF)/scripts

RESOURCE-HUTH := $(abspath $(MBDIR)/resource-huth)
HUTH-SCRIPTS := $(RESOURCE-HUTH)/scripts

RESOURCE-WILLIAMS := $(abspath $(MBDIR)/resource-williams)
WILLIAMS-SCRIPTS := $(RESOURCE-WILLIAMS)/scripts

RESOURCE-INCRSEM := $(abspath $(MBDIR)/resource-incrsem)
INCRSEM-SCRIPTS := $(RESOURCE-INCRSEM)/scripts

RESOURCE-JLSTM := $(abspath $(MBDIR)/resource-jlstm)
JLSTM-SCRIPTS := $(RESOURCE-JLSTM)/scripts

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

RESOURCE-NATSTORIESMAZE := $(abspath $(MBDIR)/resource-naturalstoriesmaze)
NATSTORIESMAZE-SCRIPTS := $(RESOURCE-NATSTORIESMAZE)/scripts

RESOURCE-NATSTORIESFMRI := $(abspath $(MBDIR)/resource-naturalstoriesfmri)
NATSTORIESFMRI-SCRIPTS := $(RESOURCE-NATSTORIESFMRI)/scripts

RESOURCE-NATSTORIESFMRI2 := $(abspath $(MBDIR)/resource-naturalstoriesfmri2)
NATSTORIESFMRI2-SCRIPTS := $(RESOURCE-NATSTORIESFMRI2)/scripts

RESOURCE-BROWN := $(abspath $(MBDIR)/resource-brown)
BROWN-SCRIPTS := $(RESOURCE-BROWN)/scripts

RESOURCE-PASSAGES := $(abspath $(MBDIR)/resource-passages)
PASSAGES-SCRIPTS := $(RESOURCE-PASSAGES)/scripts

RESOURCE-PREDRWHO := $(abspath $(MBDIR)/resource-predrwho)
PREDRWHO-SCRIPTS := $(RESOURCE-PREDRWHO)/scripts

RESOURCE-PYTHIAPILE := $(abspath $(MBDIR)/resource-pythiapile)
PYTHIAPILE-SCRIPTS := $(RESOURCE-PYTHIAPILE)/scripts

RESOURCE-ONTONOTES := $(abspath $(MBDIR)/resource-ontonotes)
ONTONOTES-SCRIPTS := $(RESOURCE-ONTONOTES)/scripts

RESOURCE-REGRESSION := $(abspath $(MBDIR)/resource-regression)
REGRESSION-SCRIPTS := $(RESOURCE-REGRESSION)/scripts

RESOURCE-RHACKS := $(abspath $(MBDIR)/resource-rhacks)
RHACKS-SCRIPTS := $(RESOURCE-RHACKS)/scripts

RESOURCE-ROARK := $(abspath $(MBDIR)/resource-roark)
ROARK-SCRIPTS := $(RESOURCE-ROARK)/scripts

RESOURCE-RT := $(abspath $(MBDIR)/resource-rt)
RT-SCRIPTS := $(RESOURCE-RT)/scripts

RESOURCE-TOKENIZE := $(abspath $(MBDIR)/resource-segment-tokenize)
TOKENIZE-SCRIPTS := $(RESOURCE-TOKENIZE)/scripts

RESOURCE-SKIPDEP := $(abspath $(MBDIR)/resource-skipdep)
SKIPDEP-SCRIPTS := $(RESOURCE-SKIPDEP)/scripts

RESOURCE-SRILM := $(abspath $(MBDIR)/resource-srilm)
SRILM-SCRIPTS := $(RESOURCE-SRILM)/scripts

RESOURCE-TOKENIZER := $(abspath $(MBDIR)/resource-tokenizer)
TOKENIZER-SCRIPTS := $(RESOURCE-TOKENIZER)/scripts

RESOURCE-TOM := $(abspath $(MBDIR)/resource-tom)
TOM-SCRIPTS := $(RESOURCE-TOM)/scripts

RESOURCE-TREEBANK := $(abspath $(MBDIR)/resource-treebank)
RESOURCE-TREEBANK-VADAS := $(abspath $(MBDIR)/resource-treebank-vadas)
TREEBANK-SCRIPTS := $(RESOURCE-TREEBANK)/scripts

RESOURCE-TIGER := $(abspath $(MBDIR)/resource-tiger)
TIGER-SCRIPTS := $(RESOURCE-TIGER)/scripts

RESOURCE-UCL := $(abspath $(MBDIR)/resource-ucl)
UCL-SCRIPTS := $(RESOURCE-UCL)/scripts

RESOURCE-UHHMM := $(abspath $(MBDIR)/resource-uhhmm)
UHHMM-SCRIPTS := $(RESOURCE-UHHMM)/scripts

RESOURCE-UPPARSE := $(abspath $(MBDIR)/resource-upparse)
UPPARSE-SCRIPTS := $(RESOURCE-UPPARSE)/scripts

RESOURCE-WORD2VEC := $(abspath $(MBDIR)/resource-word2vec)
WORD2VEC-SCRIPTS := $(RESOURCE-WORD2VEC)/scripts

RESOURCE-WORDNET := $(abspath $(MBDIR)/resource-wordnet)
WORDNET-SCRIPTS := $(RESOURCE-WORDNET)/scripts

RESOURCE-XLSX2CSV := $(abspath $(MBDIR)/resource-xlsx2csv)
XLSX2CSV-SCRIPTS := $(RESOURCE-XLSX2CSV)/scripts

RESOURCE-CONLLSCORER := $(abspath $(MBDIR)/resource-conllscorer)

BIN      := bin
GENMODEL := genmodel

define INCLUDE-ALL :=
include $(RESOURCE-TOKENIZER)/Makefile
include $(RESOURCE-LTREES)/Makefile
include $(RESOURCE-GCG)/Makefile
include $(RESOURCE-LVPCFG)/Makefile
include $(RESOURCE-SKIPDEP)/Makefile
include $(RESOURCE-EPIC)/Makefile
include $(RESOURCE-LCPARSE)/Makefile
include $(RESOURCE-BNC)/Makefile
include $(RESOURCE-TREEBANK)/Makefile
include $(RESOURCE-GIGAWORD)/Makefile
include $(RESOURCE-OPENWEBTEXT)/Makefile
include $(RESOURCE-GLOVE)/Makefile
include $(RESOURCE-GLSTM)/Makefile
include $(RESOURCE-GPT2)/Makefile
include $(RESOURCE-HUTH)/Makefile
include $(RESOURCE-KENLM)/Makefile
include $(RESOURCE-SRILM)/Makefile
include $(RESOURCE-INCRSEM)/Makefile
include $(RESOURCE-JLSTM)/Makefile
include $(RESOURCE-RT)/Makefile
include $(RESOURCE-REGRESSION)/Makefile
include $(RESOURCE-CDR)/Makefile
include $(RESOURCE-FEDECOG)/Makefile
include $(RESOURCE-FMRI)/Makefile
include $(RESOURCE-LMEFIT)/Makefile
include $(RESOURCE-LOGREG)/Makefile
include $(RESOURCE-ALICE)/Makefile
include $(RESOURCE-BMMM)/Makefile
include $(RESOURCE-CCL)/Makefile
include $(RESOURCE-CHILDES)/Makefile
include $(RESOURCE-DUNDEE)/Makefile
include $(RESOURCE-PROVO)/Makefile
include $(RESOURCE-GECO)/Makefile
include $(RESOURCE-REPETITION)/Makefile
include $(RESOURCE-CONLEN)/Makefile
include $(RESOURCE-TOM)/Makefile
include $(RESOURCE-DMV)/Makefile
include $(RESOURCE-NATSTORIES)/Makefile
include $(RESOURCE-NATSTORIESMAZE)/Makefile
include $(RESOURCE-NATSTORIESFMRI)/Makefile
include $(RESOURCE-NATSTORIESFMRI2)/Makefile
include $(RESOURCE-BROWN)/Makefile
include $(RESOURCE-PASSAGES)/Makefile
include $(RESOURCE-ONTONOTES)/Makefile
include $(RESOURCE-RHACKS)/Makefile
include $(RESOURCE-XLSX2CSV)/Makefile
include $(RESOURCE-UCL)/Makefile
include $(RESOURCE-UPPARSE)/Makefile
include $(RESOURCE-WILLIAMS)/Makefile
include $(RESOURCE-WORD2VEC)/Makefile
include $(RESOURCE-WORDNET)/Makefile
include $(RESOURCE-COREF-CONLL)/Makefile
include $(RESOURCE-CONLLSCORER)/Makefile
include $(RESOURCE-PREDRWHO)/Makefile
include $(RESOURCE-HF)/Makefile
include $(RESOURCE-PYTHIAPILE)/Makefile
endef

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-cflags.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CFLAGS := $(shell cat $(CONFIG))

CONFIG := $(CONFIGDIR)/user-numthreads.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
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

scripts:
	if [ ! -d $@ ]; then mkdir $@; fi

%config:
	if [ ! -d $@ ]; then mkdir $@; fi
