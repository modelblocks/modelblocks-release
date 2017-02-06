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

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-kenlm-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
KENLMDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(KENLMDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(KENLMDIR)$(MSG3))
endif

CONFIG := $(CONFIGDIR)/user-kenlm-model-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
KENLMMODDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(KENLMMODDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(KENLMMODDIR)$(MSG3))
endif
endif

%/user-kenlm-directory.txt: | %
	echo '.' > $@

%/user-kenlm-directory.txt.cocomo: | %
	echo '/home/compling/kenlm_git' > $(basename $@)

%/user-kenlm-model-directory.txt: | %
	echo '.' > $@
  
%/user-kenlm-model-directory.txt.cocomo: | %
	echo '/home/compling/kenlm_models' > $(basename $@)
  
  

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
%.kenlm: $(CONFIGDIR)/user-kenlm-directory.txt $(KENLMDIR)/bin/lmplz \
$$(basename %).linetoks.bz2
	mkdir kenlmtmp
	$(word 2,$^) -o $(subst .,,$(suffix $*)) -T kenlmtmp -S 50% < $(word 3,$^) > $@
	rm -rf kenlmtmp
    
.PRECIOUS: %.fw.kenlm
# especially good for unigram models since kenlm doesn't permit those to be made into binaries
%.fw.kenlm: $(KENLMDIR)/bin/lmplz $$(basename %).linetoks.bz2 \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$< -o $(subst .,,$(suffix $*)) -T kenlmtmp -S 50% < $(word 2,$^) > $@
	rm -rf kenlmtmp

.PRECIOUS: %.kenlm.bin
# uses array pointer compression
deadend%.kenlm.bin: $(KENLMDIR)/bin/build_binary %.kenlm \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$< -a 64 -T kenlmtmp -S 1G trie $(word 2,$^) $@
	rm -rf kenlmtmp

.PRECIOUS: genmodel/%.kenlm.tinybin
# smaller memory footprint, but slower
# uses array pointer compression and quantization
%.kenlm.tinybin: $(KENLMDIR)/bin/build_binary %.kenlm \
$(CONFIGDIR)/user-kenlm-directory.txt | genmodel
	mkdir kenlmtmp
	$< -a 64 -q 8 -b 8 -T kenlmtmp -S 1G trie $(word 2,$^) $@
	rm -rf kenlmtmp

.PRECIOUS: %.kenlm.ngram.itemmeasures
%.kenlm.ngram.itemmeasures: $(KENLMDIR)/bin/query \
$$(basename %).lineitems $(KENLM-SCRIPTS)/calcngramprobtoks.kenlm.py  \
$(KENLMMODDIR)/gigaword4$$(suffix $$*).kenlm.bin \
$(RT-SCRIPTS)/rename_cols.py $(CONFIGDIR)/user-kenlm-model-directory.txt | genmodel
	$< -\n $(word 4,$^) < $(word 2,$^) | python3 $(word 3,$^) --fwprobs - | \
	python $(word 5, $^) fwprob fwprob$(subst .,,$(suffix $*))  > $@

.PRECIOUS: %.multikenlm.ngram.itemmeasures
%.multikenlm.ngram.itemmeasures: %.2.kenlm.ngram.itemmeasures \
%.3.kenlm.ngram.itemmeasures %.4.kenlm.ngram.itemmeasures \
%.5.kenlm.ngram.itemmeasures $(RT-SCRIPTS)/filter_cols.py
	paste -d' ' $< <(cat $(word 2, $^) | python $(word 5, $^) -c fwprob3) \
	<(cat $(word 3, $^) | python $(word 5, $^) -c fwprob4) \
	<(cat $(word 4, $^) | python $(word 5, $^) -c fwprob5) > $@

