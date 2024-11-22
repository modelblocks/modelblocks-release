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

RESOURCE-NAME := wordfreq
define RESOURCE-DESCR =

NAME: wordfreq
MB POINTER FILE: N/A (distributed on pip)
AVAILABILITY: Free
URL: https://github.com/rspeer/wordfreq
DESCRIPTION: Word frequencies in many languages
endef

# Install with:
# pip3 install wordfreq
# pip install wordfreq[cjk]

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

#%.enunigram.itemmeasures: $(WORDFREQ-SCRIPTS)/unigram.py %.sentitems | genmodel
#	python3 $(word 1, $^) en <( cat $(word 2, $^) | grep -v '!ARTICLE' ) > $@

LANGS := ar bn bs bg ca zh hr cs da nl en fi fr de el he hi hu is id it ja ko lv lt mk ms nb fa pl pt ro ru sk sl sr es sv fil ta tr uk ur vi

define build_multiling_unigram
%.$(1)unigram.itemmeasures: $(WORDFREQ-SCRIPTS)/unigram.py %.sentitems | genmodel
	python3 $$(word 1, $$^) $(1) <( cat $$(word 2, $$^) | grep -v '!ARTICLE' ) > $$@
endef

$(foreach i,$(LANGS),\
	$(eval $(call build_multiling_unigram,$(i)))\
)
