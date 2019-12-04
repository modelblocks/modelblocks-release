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


ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-ontonotes-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
ONTONOTESDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(ONTONOTESDIR))))
RESOURCE-NAME := OntoNotes
define RESOURCE-DESCR =

NAME: OntoNotes
MB POINTER FILE: config/user-ontonotes-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/ldc2013t19
DESCRIPTION: Syntactic and semantic annotations of a large corpus comprising
various genres of text.
endef
ONTONOTES-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-ONTONOTES = $(error $(ONTONOTES-ERROR))
endif
endif

#### directory of ccgbank


%/user-ontonotes-directory.txt: | %   
	echo '/data/corpora/original/english/ontonotes5' > $@

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

#genmodel/ontowsj%.coref: $$(DEP-ONTONOTES) $(ONTONOTES-SCRIPTS)/coref2shortcoref.pl $(CONFIGDIR)/user-ontonotes-directory.txt \
$(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$* \
$$(sort $$(wildcard $(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$*/*.coref))  |  genmodel
#	cat $(wordlist 4, $(words $^), $^)  |  perl -p  $<  >  $@

# Currently fixed directory to /home/corpora/original/english/ontonotes5
genmodel/ontowsj%.name: $$(DEP-ONTONOTES) $(ONTONOTES-SCRIPTS)/name2shortname.pl $(CONFIGDIR)/user-ontonotes-directory.txt \
$(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$* \
$$(sort $$(wildcard $(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$*/*.name))  |  genmodel
	cat $(wordlist 4, $(words $^), $^)  |  perl -p  $<  >  $@

# Currently fixed to concatenate .parse files that have corresponding .name files instead of .coref files
# Change target from ontowsj -> namewsj
genmodel/ontowsj%.editabletrees: $$(DEP-ONTONOTES) $(CONFIGDIR)/user-ontonotes-directory.txt \
$(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$* $(ONTONOTES-SCRIPTS)/addArticleDelims.py  \
$$(sort $$(subst .name,.parse,$$(wildcard $(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$*/*.name)))  |  genmodel
	tail -n +1 $(wordlist 4, $(words $^), $^)  |  python $(word 3,$^)  >  $@

#genmodel/corontowsj%.editabletrees: $$(DEP-ONTONOTES) $(CONFIGDIR)/user-ontonotes-directory.txt \
$(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$* $(ONTONOTES-SCRIPTS)/addArticleDelims.py  \
$$(sort $$(subst .coref,.parse,$$(wildcard $(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$*/*.coref)))  |  genmodel
#	tail -n +1 $(wordlist 4, $(words $^), $^)  |  python $(word 3,$^)  >  $@

%.linked.linetrees:  $(ONTONOTES-SCRIPTS)/coref2linkedlinetrees.py  $$(dir %)$$(word 1,$$(subst ., ,$$*)).coref  %.linetrees
	#cat $(word 2,$^)  |  python $(word 1,$^)  $(word 3,$^)  >  $@
	python $(word 1,$^)  $(word 3,$^)  $(word 2,$^)  >  $@

# editing...
%.ner.linetrees:  $(ONTONOTES-SCRIPTS)/name2nerlinetrees.py  $$(dir %)$$(word 1,$$(subst ., ,$$*)).name  %.linetrees
	#cat $(word 2,$^)  |  python $(word 1,$^)  $(word 3,$^)  >  $@
	python $(word 1,$^)  $(word 3,$^)  $(word 2,$^)  >  $@

%.unlinked.linetrees: %.linetrees
	cat $<  |  perl -pe 's/[-+]n[0-9]+(?=[^\)]* )//g'  >  $@

#### collections
#genmodel/ontowsj02to21.%.linked.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,genmodel/ontowsj$(sect).%.linked.linetrees)
#	cat $^  >  $@

genmodel/corontowsj%.coref: $$(DEP-ONTONOTES) $(ONTONOTES-SCRIPTS)/coref2shortcoref.pl $(CONFIGDIR)/user-ontonotes-directory.txt \
$(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$* \
$$(sort $$(wildcard $(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$*/*.coref))  |  genmodel
	cat $(wordlist 4, $(words $^), $^)  |  perl -p  $<  >  $@

genmodel/corontowsj%.editabletrees: $$(DEP-ONTONOTES) $(CONFIGDIR)/user-ontonotes-directory.txt \
$(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$* $(ONTONOTES-SCRIPTS)/addArticleDelims.py  \
$$(sort $$(subst .coref,.parse,$$(wildcard $(ONTONOTESDIR)/data/files/data/english/annotations/nw/wsj/$$*/*.coref)))  |  genmodel 
	tail -n +1 $(wordlist 4, $(words $^), $^)  |  python $(word 3,$^)  >  $@

.PRECIOUS: genmodel/ontowsj02to21.%.linetrees
genmodel/ontowsj02to21.%.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,genmodel/ontowsj$(sect).%.linetrees)
	cat $^  >  $@

.PRECIOUS: genmodel/ontowsjdev.%.linetrees 
genmodel/ontowsjdev.%.linetrees: $(foreach sect,01 22 24,genmodel/ontowsj$(sect).%.linetrees)
	cat $^  >  $@

.PRECIOUS: genmodel/conllwsjdev.%.linetrees
genmodel/conllwsjdev.%.linetrees: $(foreach sect,00 01 22 24,genmodel/ontowsj$(sect).%.linetrees)
	cat $^  >  $@

#.PRECIOUS: genmodel/corontowsj02to21.%.linetrees
#genmodel/corontowsj02to21.%.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,genmodel/corontowsj$(sect).%.linetrees)
#	cat $^  >  $@
.PRECIOUS: genmodel/corontowsj02to21.casp.prtrm.linked.linetrees
genmodel/corontowsj02to21.casp.prtrm.linked.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,genmodel/corontowsj$(sect).casp.prtrm.linked.linetrees)
	cat $^  >  $@

#.PRECIOUS: genmodel/corontowsjdev.%.linetrees 
#genmodel/corontowsjdev.%.linetrees: $(foreach sect,01 22 24,genmodel/corontowsj$(sect).%.linetrees)
#	cat $^  >  $@
.PRECIOUS: genmodel/corontowsjdev.casp.prtrm.linked.linetrees 
genmodel/corontowsjdev.casp.prtrm.linked.linetrees: $(foreach sect,01 22 24,genmodel/corontowsj$(sect).casp.prtrm.linked.linetrees)
	cat $^  >  $@

.PRECIOUS: genmodel/corconllwsjdev.casp.prtrm.linked.linetrees
genmodel/corconllwsjdev.casp.prtrm.linked.linetrees: $(foreach sect,00 01 22 24,genmodel/corontowsj$(sect).casp.prtrm.linked.linetrees)
	cat $^  >  $@

#.PRECIOUS: genmodel/corconllwsjdev.casp.linked.linetrees
#genmodel/corconllwsjdev.casp.linked.linetrees: $(foreach sect,00 01 22 24,genmodel/corontowsj$(sect).casp.linked.linetrees)
#	cat $^  >  $@

#genmodel/ontowsj%.linked.linetrees:  user-ontonotes-directory.txt  scripts/ments2linkedtrees.py  $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*
#	echo '' > $@
#	for file in $(word 3,$^)/*.coref;  do python $(word 2,$^) $$file >> $@;  done

#%.linked.linetrees:  %.linetrees  scripts/ments2linktags.py  user-ontonotes-directory.txt  $$(shell cat user-ontonotes-directory.txt)/ontonotes-4.0/data/files/data/english/annotations/nw/wsj/$$*
#	cat $<  |  $(word 2,$^) 

