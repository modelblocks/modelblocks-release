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
#  i. Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

#The directory variables break when this makefile is called by another.
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SRCMODEL := $(THISDIR)srcmodel
SCRIPTS  := $(THISDIR)scripts
WINDOWLEN := 1

#GloVe internal parameters
VERBOSE := 2
MEMORY := 4.0
VOCAB_MIN_COUNT := 5
MAX_ITER := 15
WINDOW_SIZE := 15
BINARY := 0
NUM_THREADS := 8
X_MAX := 100

################################################################################
#
#  ii. User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

.PRECIOUS: %config/
%config/:
	mkdir $*config/

#### Create user-*.txt file for resource at include time, prior to make-item dependency graph construction...
ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-glove-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG))) 
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GLOVEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GLOVEDIR))))
RESOURCE-NAME := GloVe
define RESOURCE-DESCR =

NAME: GloVe
MB POINTER FILE: config/user-glove-directory.txt
AVAILABILITY: Free
URL: https://nlp.stanford.edu/projects/glove/
DESCRIPTION: GloVe is an unsupervised learning algorithm for obtaining vector representations
for words. Training is performed on aggregated global word-word co-occurrence statistics from
a corpus, and the resulting representations showcase interesting linear substructures of the
word vector space.

These recipes automate training of GloVe vectors on arbitrary text data.

They also support the use of embeddings pre-trained on the 840B Common Crawl dataset, which can be
downloaded at the URL above.
endef
GLOVE-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GLOVE = $(error $(GLOVE-ERROR))
endif
endif



%/user-glove-directory.txt: | % 
	echo '/fs/project/lin-dept/compling/GloVe-1.2/' > $@

################################################################################
#
#  1. Wordnet items
#
################################################################################

%.CC.300.glove.embeddings.tokmeasures: $$(DEP-GLOVE) %.linetoks $(SCRIPTS)/embedDocument.py $(GLOVEDIR)glove.840B.300d.txt
	cat $(word 1, $^) | python $(word 2, $^) $(word 3, $^) -s > $@

%.CC.300.glove.linemeasures: $$(DEP-GLOVE) %.linetoks $(SCRIPTS)/embedDocument.py $(GLOVEDIR)glove.840B.300d.txt
	cat $(word 1, $^) | python $(word 2, $^) $(word 3, $^) -s -o linemeasures -w $(WINDOWLEN) > $@

%.CC.300.glove.tokmeasures: $$(DEP-GLOVE) %.linetoks $(SCRIPTS)/embedDocument.py $(GLOVEDIR)glove.840B.300d.txt
	cat $(word 1, $^) | python $(word 2, $^) $(word 3, $^) -s -o tokmeasures -w $(WINDOWLEN) > $@

%.glove.embeddings.tokmeasures: $$(basename $$(basename %)).linetoks $(SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename %)))$$(suffix $$*).glove.txt
	cat $(word 1, $^) | python $(word 2, $^) $(word 3, $^) -s > $@

%.glove.linemeasures: $$(basename $$(basename $$(basename %))).linetoks $(SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename $$(basename %))))$$(suffix $$(basename $$*)).glove.txt
	cat $(word 1, $^) | python $(word 2, $^) $(word 3, $^) -s -o linemeasures -w $(subst _,  ,$(subst .,,$(suffix $*))) > $@

%.glove.tokmeasures: $$(basename $$(basename $$(basename %))).linetoks $(SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename $$(basename %))))$$(suffix $$(basename $$*)).glove.txt
	cat $(word 1, $^) | python $(word 2, $^) $(word 3, $^) -s -o tokmeasures -w $(subst _, ,$(subst .,,$(suffix $*))) > $@

%.vocab.txt: $$(DEP-GLOVE) %.linetoks $(GLOVEDIR)build/vocab_count
	$(word 2, $^) -min-count $(VOCAB_MIN_COUNT) -verbose $(VERBOSE) < $(word 1, $^) > $@

%.cooccurrence.bin: $$(DEP-GLOVE) %.linetoks %.vocab.txt $(GLOVEDIR)build/cooccur
	$(word 3, $^) -m $(MEMORY) -vocab-file $(word 2, $^) -verbose $(VERBOSE) -window-size $(WINDOW_SIZE) < $(word 1, $^) > $@

%.cooccurrence.shuf.bin: $$(DEP-GLOVE) %.cooccurrence.bin $(GLOVEDIR)build/shuffle
	$(word 2, $^) -memory $(MEMORY) -verbose $(VERBOSE) < $(word 1, $^) > $@

.PRECIOUS: %.glove.txt
%.glove.txt: $$(DEP-GLOVE) $$(basename %).vocab.txt $$(basename %).cooccurrence.shuf.bin $(GLOVEDIR)build/glove
	$(word 3, $^) -save-file $(basename $@) -threads $(NUM_THREADS) -input-file $(word 2, $^) -x-max $(X_MAX) -iter $(MAX_ITER) -vector-size $(subst .,,$(suffix $*)) -binary $(BINARY) -vocab-file $(word 1, $^) -verbose $(VERBOSE)

