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
#  include $(RESOURCE-TREEBANK)/Makefile
#  include $(RESOURCE-TOKENIZER)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

define ParamVal
SRILM:
  kernel: NgramModel
  text: SRILM
  value: srilm
endef

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
CONFIG := $(CONFIGDIR)/user-srilm-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
SRILMDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(SRILMDIR))))
RESOURCE-NAME := SRILM Language Model Toolkit
define RESOURCE-DESCR =

NAME: SRILM Language Model Toolkit
MB POINTER FILE: config/user-srilm-directory.txt
AVAILABILITY: Free for non-commercial use
URL: http://www.speech.sri.com/projects/srilm/download.html
DESCRIPTION: SRILM is a toolkit for building and applying statistical language models (LMs),
primarily for use in speech recognition, statistical tagging and segmentation,
and machine translation.
endef
SRILM-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-SRILM = $(error $(SRILM-ERROR))
else
DEP-SRILM :=
endif
endif



%/user-srilm-directory.txt: | %
	echo '/home/compling/srilm/bin/i686-m64' > $@

################################################################################
#
#  Ngram recipes
#
################################################################################

broadcoveragetraining.%.linetoks: $$(DEP-TOKENIZER) $(TOKENIZERDIR) \
genmodel/brownTRAIN.linetoks genmodel/wsj02to21.linetoks genmodel/bncTRAIN.linetoks %.linetoks \
$(CONFIGDIR)/user-tokenizer-directory.txt | genmodel
	cat $(word 4,$^) | sed "s/&bquo;/\'/g;s/&equo;/\'/g;s/&hellip;/.../g;s/&percnt;/%/g;s/&ndash;/-/g;s/&amp;/\&/g;s/&mdash;/--/g;" | \
	$(word 1, $^)/ptb_tokenizer.sed > genmodel/tmp.bnc
	$(word 1, $^))/ptb_tokenizer.sed $(word 5,$^) > genmodel/tmp.$(notdir $*)
	cat $(word 2,$^) $(word 3,$^) genmodel/tmp.bnc genmodel/tmp.$(notdir $*) | grep -v '\*x\*' | sed 's/  */ /g;' > $@
	rm -f genmodel/tmp.bnc genmodel/tmp.$(notdir $*)
    
broadcoveragetraining.%.revsents: broadcoveragetraining.%.linetoks | genmodel
	cat $^ | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/ <LINEBREAK> /g' > $@.tmp #the idiocy of BSD sed (e.g. on OSX) requires the dumb formulation of this sed command
	tac -s ' ' $@.tmp | sed 's/ <LINEBREAK> /\n/g;s/  */ /g;' > $@
	rm -f $@.tmp

%.srilmngrams: $$(DEP-SRILM) %.linetoks %.revsents $(SRILM-SCRIPTS)/convert_srilm.py $(SRILMDIR) $(CONFIGDIR)/user-srilm-directory.txt
	$(word 4, $^)/ngram-count -order 1 -kndiscount -text $< -lm $@.uprobs
	$(word 4, $^)/ngram-count -order 2 -kndiscount -interpolate -text $< -lm $@.fwprobs
	$(word 4, $^)/ngram-count -order 2 -kndiscount -interpolate -text $(word 2, $^) -lm $@.bwprobs
	python3 $(word 3,$^) $@.uprobs -U > $@
	python3 $(word 3,$^) $@.fwprobs -BF >> $@
	python3 $(word 3,$^) $@.bwprobs -BB >> $@
	rm -f $@.{uprobs,fwtmp,fwprobs,bwtmp,bwprobs}
    
%.textdata: %.lineitems
	cat <(echo WORD) <(cat $< | sed 's/ /\n/g;') > $@
  
%.srilm.ngram.itemmeasures: $$(DEP-SRILM) $(SRILM-SCRIPTS)/calcngramprobtoks.srilm.py \
broadcoveragetraining.%.srilmngrams %.textdata | genmodel
	python3 $(word 1, $^) > $@
