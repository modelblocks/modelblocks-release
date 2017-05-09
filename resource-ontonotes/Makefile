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
#  include $(RESOURCE-GCG)/Makefile
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

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-ontonotes-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
ONTONOTESDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(CHILDESDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(CHILDESDIR)$(MSG3))
endif
endif

#### directory of ccgbank


%/user-ontonotes-directory.txt: | %   
	echo '/home/corpora/original/various/ontonotes' > $@

################################################################################
#
#  Ontonotes
#
#  to construct the following file types:
#    <x>.linked.linetrees  : phrase structure trees, one sentence per line, bracketed by parens and linked with -n tags
#
################################################################################

#genmodel/ontowsj%.cuecorefs:  scripts/coref2cuecorefs.py  user-ontonotes-directory.txt  $$(wildcard $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*/*.coref)
#	cat $(wordlist 3, $(words $^), $^)  |  python $<  >  $@

genmodel/ontowsj%.coref:  $(ONTONOTES-SCRIPTS)/coref2shortcoref.pl  user-ontonotes-directory.txt  $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*  $$(sort $$(wildcard $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*/*.coref))  |  genmodel
	cat $(wordlist 4, $(words $^), $^)  |  perl -p  $<  >  $@

genmodel/ontowsj%.editabletrees:  user-ontonotes-directory.txt  $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*  $$(sort $$(subst .coref,.parse,$$(wildcard $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*/*.coref)))  |  genmodel
	tail -n +1 $(wordlist 3, $(words $^), $^)  |  perl -pe 's/==>.*<==/(U !ARTICLE)/'  >  $@

%.linked.linetrees:  $(ONTONOTES-SCRIPTS)/coref2linkedlinetrees.py  $$(dir %)$$(word 1,$$(subst ., ,$$*)).coref  %.linetrees
	cat $(word 2,$^)  |  python $(word 1,$^)  $(word 3,$^)  >  $@

#### collections
genmodel/ontowsj02to21.%.linked.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,genmodel/ontowsj$(sect).%.linked.linetrees)
	cat $^  >  $@


#genmodel/ontowsj%.linked.linetrees:  user-ontonotes-directory.txt  scripts/ments2linkedtrees.py  $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*
#	echo '' > $@
#	for file in $(word 3,$^)/*.coref;  do python $(word 2,$^) $$file >> $@;  done

#%.linked.linetrees:  %.linetrees  scripts/ments2linktags.py  user-ontonotes-directory.txt  $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*
#	cat $<  |  $(word 2,$^) 

