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
MB POINTER FILE: config/user-kenlm-directory.txt
AVAILABILITY: Free
URL: https://kheafield.com/code/kenlm/
DESCRIPTION: KenLM estimates, filters, and queries language models. Estimation
is fast and scalable due to streaming algorithms.
endef
KENLM-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-KENLM = $(error $(KENLM-ERROR))
else
DEP-KENLM :=
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
MB POINTER FILE: config/user-kenlm-model-directory.txt
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
DEP-KENLM :=
endif

CONFIG := $(CONFIGDIR)/user-kenlm-flags.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
KENLMFLAGS := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(KENLMFLAGS))))
RESOURCE-NAME := KenLM Language Model Toolkit (runtime flags)
define RESOURCE-DESCR =

NAME: KenLM Language Model Toolkit (runtime flags)
MB POINTER FILE: config/user-kenlm-flags.txt
AVAILABILITY: Free
URL: https://kheafield.com/code/kenlm/
DESCRIPTION: KenLM estimates, filters, and queries language models. Estimation
is fast and scalable due to streaming algorithms.

This resource is a holder for runtime flags used with kenlm on your system.
You can revise the flags using the file above.
endef
KENLMFLAGS-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-KENLMFLAGS = $(error $(KENLMFLAGS-ERROR))
else
DEP-KENLM :=
endif
endif



%/user-kenlm-directory.txt: | %
	echo '/fs/project/lin-dept/compling/kenlm_git' > $@

%/user-kenlm-model-directory.txt: | %
	echo '/fs/project/lin-dept/compling/kenlm_models' > $@

%/user-kenlm-flags.txt: | %
	echo '-T kenlmtmp -S 50%' > $@

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
$$(basename %).linetoks.bz2 $(CONFIGDIR)/user-kenlm-flags.txt
	mkdir kenlmtmp
	$(word 2,$^) -o $(subst .,,$(suffix $*)) $(shell cat $(word 4,$^)) --skip_symbols < $(word 3, $^) > $@
	rm -rf kenlmtmp

#.PRECIOUS: %.fw.kenlm
## especially good for unigram models since kenlm doesn't permit those to be made into binaries
#%.fw.kenlm: $$(DEP-KENLM) $(KENLMDIR)/bin/lmplz $$(basename %).linetoks.bz2 \
#$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
#	mkdir kenlmtmp
#	$(word 1, $^) -o $(subst .,,$(suffix $*)) -T kenlmtmp -S 20G --skip_symbols < $(word 2, $^) > $@
#	rm -rf kenlmtmp

.PRECIOUS: %.kenlm.bin
# uses array pointer compression
deadend%.kenlm.bin: $$(DEP-KENLM) $(KENLMDIR)/bin/build_binary %.kenlm \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$(word 1, $^) -a 64 -T kenlmtmp -S 1G trie $(word 2, $^) $@
	rm -rf kenlmtmp

.PRECIOUS: %.kenlm.tinybin
# smaller memory footprint, but slower
# uses array pointer compression and quantization
%.kenlm.tinybin: $$(DEP-KENLM) $(KENLMDIR)/bin/build_binary %.kenlm \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$(word 1, $^) -a 64 -q 8 -b 8 -T kenlmtmp -S 1G trie $(word 2, $^) $@
	rm -rf kenlmtmp


## TO SPEC
.PRECIOUS: %-kenlm.itemmeasures
%-kenlm.itemmeasures: $$(DEP-KENLM) $(KENLMDIR)/bin/query \
$$(basename %).sentitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$(KENLMMODDIR)/gigaword$$(suffix $$*).kenlm.bin \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	cat $(word 2,$^)  |  grep -v '!ARTICLE'  |  $(word 1, $^) -\n $(word 4,$^)  |  python3 $(word 3,$^) --fwprobs -  |  \
	python3 $(word 5, $^) fwprob fwprob$(subst .,,$(suffix $*))  >  $@

%-kenlm.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLMDIR)/bin/query \
$$(basename %).sentitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$(KENLMMODDIR)/gigaword$$(suffix $$*).kenlm \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	$(word 1, $^) -\n $(word 4,$^) <( cat $(word 2,$^) | grep -v '~ARTICLE' ) | python3 $(word 3,$^) --fwprobs - | \
	python3 $(word 5, $^) fwprob fwprob$(subst .,,$(suffix $*))  > $@

%-kenlmlocal.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLMDIR)/bin/query \
$$(basename %).sentitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$$(subst -,.,$$(subst .,,$$(suffix %))).kenlm.tinybin \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	cat $(word 2,$^)  |  grep -v '!ARTICLE'  |  $(word 1, $^) -\n $(word 4,$^)  |  python3 $(word 3,$^) --fwprobs -  |  \
	python3 $(word 5, $^) fwprob fwprob$(subst -,,$(subst .,,$(suffix $*)))  > $@

%.unigram.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLM-SCRIPTS)/unigram.py $(KENLMMODDIR)/gigaword.1.fw.kenlm %.sentitems \
$(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	python3 $(word 1, $^) $(word 2, $^) <( cat $(word 3, $^) | grep -v '!ARTICLE' ) > $@

%.owtunigram.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLM-SCRIPTS)/unigram.py $(KENLMMODDIR)/openwebtext.1.fw.kenlm %.sentitems \
$(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	python3 $(word 1, $^) $(word 2, $^) <( cat $(word 3, $^) | grep -v '!ARTICLE' ) > $@

%-unigram.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLM-SCRIPTS)/unigram.py $$(subst .,,$$(suffix %)).1.fw.kenlm $$(basename %).sentitems \
$(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	python3 $(word 1, $^) $(word 2, $^) <( cat $(word 3, $^) | grep -v '!ARTICLE' ) > $@

## not to spec
.PRECIOUS: %.kenlm.ngram.itemmeasures
%.kenlm.ngram.itemmeasures: $$(DEP-KENLM) $(KENLMDIR)/bin/query \
$$(basename %).lineitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$(KENLMMODDIR)/gigaword$$(suffix $$*).kenlm.bin \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	$(word 1, $^) -\n $(word 4,$^) < $(word 2,$^) | python3 $(word 3,$^) --fwprobs - | \
	python3 $(word 5, $^) fwprob fwprob$(subst .,,$(suffix $*))  > $@

%.kenlm.ngram.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLMDIR)/bin/query \
$$(basename %).lineitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$(KENLMMODDIR)/gigaword$$(suffix $$*).kenlm \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	$(word 1, $^) -\n $(word 4,$^) < $(word 2,$^) | python3 $(word 3,$^) --fwprobs - | \
	python3 $(word 5, $^) fwprob fwprob$(subst .,,$(suffix $*))  > $@

.PRECIOUS: %.multikenlm.ngram.itemmeasures
%.multikenlm.ngram.itemmeasures: %.2.kenlm.ngram.itemmeasures \
%.3.kenlm.ngram.itemmeasures %.4.kenlm.ngram.itemmeasures \
%.5.kenlm.ngram.itemmeasures $(RT-SCRIPTS)/filter_cols.py
	paste -d' ' $< <(cat $(word 2, $^) | python3 $(word 5, $^) -c fwprob3) \
	<(cat $(word 3, $^) | python3 $(word 5, $^) -c fwprob4) \
	<(cat $(word 4, $^) | python3 $(word 5, $^) -c fwprob5) > $@

.PRECIOUS: %.multishortkenlm.ngram.itemmeasures
%.multishortkenlm.ngram.itemmeasures: %.2.kenlm.ngram.itemmeasures \
%.3.kenlm.ngram.itemmeasures %.4.kenlm.ngram.itemmeasures \
$(RT-SCRIPTS)/filter_cols.py
	paste -d' ' $< <(cat $(word 2, $^) | python3 $(word 4, $^) -c fwprob3) \
	<(cat $(word 3, $^) | python3 $(word 4, $^) -c fwprob4) > $@

%.kenlm.unigram.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLM-SCRIPTS)/unigram.py $(KENLMMODDIR)/gigaword.1.fw.kenlm %.lineitems \
$(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	echo $(dir, $*)/$(suffix $*).1.fw.kenlm
	python3 $(word 1, $^) $(word 2, $^) $(word 3, $^) > $@

%.Kenlm.unigram.itemmeasures: $$(DEP-KENLM) $$(DEP-KENLMMOD) $(KENLM-SCRIPTS)/unigram.py $$(dir $$*)/$$(suffix $$*).1.fw.kenlm $$(basename %).lineitems \
$(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	python3 $(word 1, $^) $(word 2, $^) $(word 3, $^) > $@



