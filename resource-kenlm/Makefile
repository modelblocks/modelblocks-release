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
#  include $(RESOURCE-LTREES)/Makefile
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
CONFIG := $(CONFIGDIR)/user-kenlm-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
KENLMDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(KENLMDIR))))
RESOURCE-NAME := KenLM Language Model Toolkit
define RESOURCE-DESCR =

NAME: KenLM Language Model Toolkit
AVAILABILITY: Free
URL: https://kheafield.com/code/kenlm/
DESCRIPTION: KenLM estimates, filters, and queries language models. Estimation
is fast and scalable due to streaming algorithms.
endef
KENLM-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-KENLM = $(error $(KENLM-ERROR))
else
DEP-KENML :=
endif

CONFIG := $(CONFIGDIR)/user-kenlm-model-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
KENLMMODDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(KENLMMODDIR))))
RESOURCE-NAME := KenLM Language Model Toolkit (model binaries directory)
define RESOURCE-DESCR =

NAME: KenLM Language Model Toolkit (model binaries directory)
AVAILABILITY: Free
URL: https://kheafield.com/code/kenlm/
DESCRIPTION: KenLM estimates, filters, and queries language models. Estimation
is fast and scalable due to streaming algorithms.

This resource is just a directory in which to store compiled binaries.
You can specify a binaries directory using the pointer file above.
endef
KENLMMOD-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-KENLMMOD = $(error $(KENLMMOD-ERROR))
else
DEP-KENML :=
endif
endif



%/user-kenlm-directory.txt: | %
	echo '/home/compling/kenlm_git' > $@


  
%/user-kenlm-model-directory.txt: | %
	echo '/home/compling/kenlm_models' > $@
  
  

################################################################################
#
#  Ngram recipes
#
################################################################################
    
.PRECIOUS: %.bz2
%.bz2: %
	#careful, this removes the source after compressing a copy
	#saves space, but it makes the order of the dependencies in a make item crucial; e.g. revsents.bz2 must come before sents.bz2
	bzip2 $<
  
.PRECIOUS: %.kenlm
# especially good for unigram models since kenlm doesn't permit those to be made into binaries
%.kenlm: $$(DEP-KENLM) $(CONFIGDIR)/user-kenlm-directory.txt $(KENLMDIR)/bin/lmplz \
$$(basename %).linetoks.bz2
	mkdir kenlmtmp
	$(word 3,$^) -o $(subst .,,$(suffix $*)) -T kenlmtmp -S 50% < $(word 4,$^) > $@
	rm -rf kenlmtmp
    
.PRECIOUS: %.fw.kenlm
# especially good for unigram models since kenlm doesn't permit those to be made into binaries
%.fw.kenlm: $$(DEP-KENLM) $(KENLMDIR)/bin/lmplz $$(basename %).linetoks.bz2 \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$(word 2, $^) -o $(subst .,,$(suffix $*)) -T kenlmtmp -S 50% < $(word 3,$^) > $@
	rm -rf kenlmtmp

.PRECIOUS: %.kenlm.bin
# uses array pointer compression
deadend%.kenlm.bin: $$(DEP-KENLM) $(KENLMDIR)/bin/build_binary %.kenlm \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$(word 2, $^) -a 64 -T kenlmtmp -S 1G trie $(word 3,$^) $@
	rm -rf kenlmtmp

.PRECIOUS: genmodel/%.kenlm.tinybin
# smaller memory footprint, but slower
# uses array pointer compression and quantization
%.kenlm.tinybin: $$(DEP-KENLM) $(KENLMDIR)/bin/build_binary %.kenlm \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$(word 2, $^) -a 64 -q 8 -b 8 -T kenlmtmp -S 1G trie $(word 3,$^) $@
	rm -rf kenlmtmp

define ParamVal
KENLM:
  kernel: NgramModel
  text: KENLM
  value: kenlm
endef
define KernelBlock
NgramModel:
  blocktitle: N-gram model
  paramtype: Dropdown
NgramOrder:
  blocktitle: N-gram order
  paramtype: Integer
  paramval:
  - text: N-gram order
    value: ''
endef
define CompositeBlock
NgramParams:
  blockseq:
  - blocktype: NgramOrder
  - blocktype: NgramModel
  - blocktype: NgramIZ
  blocktitle: N-gram Parameters
endef
define TargetBlock
NgramProbs:
  blocktitle: N-gram Probabilities
  targetsuffix: .ngram.itemmeasures
  blockseq:
  - blocktype: Corpus
  - blocktype: NgramParams
    kwargs:
      innerdelim: '.'
endef
.PRECIOUS: %.kenlm.ngram.itemmeasures
%.kenlm.ngram.itemmeasures: $$(DEP-KENLM) $(KENLMDIR)/bin/query \
$$(basename %).lineitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$(KENLMMODDIR)/gigaword4$$(suffix $$*).kenlm.bin \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	$(word 2, $^) -\n $(word 5,$^) < $(word 3,$^) | python3 $(word 4,$^) --fwprobs - | \
	python $(word 6, $^) fwprob fwprob$(subst .,,$(suffix $*))  > $@

%.kenlm.ngram.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLMDIR)/bin/query \
$$(basename %).lineitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$(KENLMMODDIR)/gigaword4$$(suffix $$*).kenlm \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	$(word 3, $^) -\n $(word 6,$^) < $(word 4,$^) | python3 $(word 5,$^) --fwprobs - | \
	python $(word 7, $^) fwprob fwprob$(subst .,,$(suffix $*))  > $@

.PRECIOUS: %.multikenlm.ngram.itemmeasures
%.multikenlm.ngram.itemmeasures: %.2.kenlm.ngram.itemmeasures \
%.3.kenlm.ngram.itemmeasures %.4.kenlm.ngram.itemmeasures \
%.5.kenlm.ngram.itemmeasures $(RT-SCRIPTS)/filter_cols.py
	paste -d' ' $< <(cat $(word 2, $^) | python $(word 5, $^) -c fwprob3) \
	<(cat $(word 3, $^) | python $(word 5, $^) -c fwprob4) \
	<(cat $(word 4, $^) | python $(word 5, $^) -c fwprob5) > $@

.PRECIOUS: %.multishortkenlm.ngram.itemmeasures
%.multishortkenlm.ngram.itemmeasures: %.2.kenlm.ngram.itemmeasures \
%.3.kenlm.ngram.itemmeasures %.4.kenlm.ngram.itemmeasures \
$(RT-SCRIPTS)/filter_cols.py
	paste -d' ' $< <(cat $(word 2, $^) | python $(word 4, $^) -c fwprob3) \
	<(cat $(word 3, $^) | python $(word 4, $^) -c fwprob4) > $@

