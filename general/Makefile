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

.SECONDARY:
.SUFFIXES:
.SECONDEXPANSION:

THISDIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRC      := $(THISDIR)src
SCRIPTS  := $(THISDIR)scripts
SRCMODEL := $(THISDIR)srcmodel

BIN      := bin
GENMODEL := genmodel

################################################################################
#
#  ii. User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

#### c++ compile flags
user-cflags.txt:   ## -g
	echo '-DNDEBUG -O3' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may not be to your liking'
	@echo 'edit it to tell C++ whether to compile in debug mode or optimize, and re-run make to continue!'
	@echo ''

#### directory of treebank
user-treebank-directory.txt:
	echo '/home/corpora/original/english/penn_treebank_3' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your treebank repository, and re-run make to continue!'
	@echo ''

################################################################################
#
#  iii. Code compilation items
#
################################################################################

# CODE REVIEW: some of these are bogus!!!!
SHELL = /bin/bash
#INCLUDES = -I../rvtl/include #-Iinclude -I../slush/include #-I/sw/include #-I/Users/dingcheng/Documents/boost/boost_1_44_0
#CFLAGS = -Wall `cat user-cflags.txt` -g #-DNDEBUG -O3 #-DNOWARNINGS #-g #
#CC = g++

#### bin directory (ignored by git b/c empty)
bin:
	if [ ! -d $@ ]; then mkdir $@; fi


bin/evalb: $(SRC)/evalb.c | bin
	gcc -Wall -g -o $@ $<

#### c++ executables
.PRECIOUS: bin/%
bin/%:  $(SRC)/$$(notdir $$@).cpp  user-cflags.txt  |  bin  #$$(abspath $$(dir $$@))  #$$(realpath $$(dir $$@))  #$(BIN)  #$(CURDIR)/$$(notdir $$(dir $$@))  #
	g++ -I../rvtl/include -Wall `cat $(word 2,$^)` -g -lm  $<  -o $@

################################################################################
#
#  1. Printout
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
#  1. Penn Treebank
#
#  to construct the following file types:
#    <x>.linetrees : treebank-tagset phase structure trees, one sentence per line, bracketed by parens
#
################################################################################

#### genmodel directory (ignored by git b/c empty)
#$(GENMODEL):
genmodel:
	if [ ! -d $@ ]; then mkdir $@; fi

#### each wsj## file
.PRECIOUS: genmodel/wsj%.linetrees
genmodel/wsj%.linetrees:  user-treebank-directory.txt  $$(shell cat user-treebank-directory.txt)/parsed/mrg/wsj/$$*  $(SCRIPTS)/editabletrees2linetrees.pl  |  genmodel  #$$(abspath $$(dir $$@))
	cat $(word 2,$^)/*.mrg  |  perl $(word 3,$^)  >  $@

#### obtain (flat) linetrees from (indented) editabletrees
%.linetrees:  %.editabletrees  $(SCRIPTS)/editabletrees2linetrees.pl
	cat $<  |  perl -pe 's/^[0-9]*://'  |  perl $(word 2,$^)  >  $@
#%.editabletrees:  %.linetrees  bin/indent
#	cat $<  |  $(word 2,$^)  >  $@

#### generalized operations on linetrees
%.linetrees:  $(SCRIPTS)/make-trees-$$(subst .,,$$(suffix $$*)).pl  $$(basename %).linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@

#### collections
%/wsj02to21.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,%/wsj$(sect).linetrees)
	cat $^  >  $@

#### use only the first/last N trees
%first.linetrees:  $$(basename %).linetrees
	head -$(subst .,,$(suffix $*)) $<  >  $@
%onward.linetrees: $$(basename %).linetrees
	tail -n +$(subst .,,$(suffix $*)) $<  >  $@
%last.linetrees:  $$(basename %).linetrees
	tail -$(subst .,,$(suffix $*)) $<  >  $@
%maxwords.linetrees: $$(basename %).linetrees
	cat $< | perl -na -e "if (split(/[\)]+/)<=1+$(subst .,,$(suffix $*))) {print $$_;}"  >  $@ 

#### obtain input sentences from linetrees
.PRECIOUS: %.sents
%.sents:  $(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(notdir $$*.linetrees)))
	cat $<  |  sed 's/(-NONE-[^)]*)//g'  \
		|  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$$//;'  \
		|  sed 's/!unf! *//g' >  $@
%.sents:  %.linetrees
	cat $<  |  sed 's/(-NONE-[^)]*)//g'  \
		|  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$$//;'  \
		|  sed 's/!unf! *//g' >  $@

#### unlabeled trees (for evaluation)
%.nolabel.linetrees:  %.linetrees
	cat $^  |  perl -pe 's/\(\)//'| perl -pe 's/\([^ ]*/\(X/g'  >  $@

%.nolabel.nopuncE.linetrees:%.nolabel.linetrees
	cat $<  |  perl -pe "s/ \(X (,|\.|``|--|-RRB-|-LRB-|-LCB-|-RCB-|''|\.\.\.|\?|\:|\;)\)//g" | perl ../general/scripts/make-trees-nounary.pl > $@

#### use only the first/last N trees
%first.sents:  $$(basename %).sents
	head -$(subst .,,$(suffix $*)) $<  >  $@
%onward.sents: $$(basename %).sents
	tail -n +$(subst .,,$(suffix $*)) $<  >  $@
%last.sents:  $$(basename %).sents
	tail -$(subst .,,$(suffix $*)) $<  >  $@

#### rules
.PRECIOUS: %.rules
%.rules: %.linetrees  $(SCRIPTS)/trees2rules.pl
	cat $<  |  perl $(word 2,$^)  >  $@

#### syntactic evaluation           #genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees  %.linetrees
%.syneval:  bin/evalb \
            $(SRCMODEL)/new.prm \
            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).linetrees
	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@
#### syntactic evaluation for nounary post-process
%.nounary.syneval:  bin/evalb \
            $(SRCMODEL)/new.prm \
            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).nounary.linetrees
	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@
#### syntactic evaluation for nounary nolabel post-process
%.nounary.nolabel.syneval:  bin/evalb \
            $(SRCMODEL)/new.prm \
            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).nounary.nolabel.linetrees
	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@

#### syntactic evaluation for nounary nolabel nopunctuation post-process
%.nounary.nolabel.nopuncE.syneval:  bin/evalb \
            $(SRCMODEL)/new.prm \
            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).nounary.nolabel.nopuncE.linetrees
	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@

#### signif testing
%.bootstrapsignif:  $(SCRIPTS)/compare.pl \
         $$(word 1,$$(subst .., ,%)).$$(word 2,$$(subst .., ,$$*)).$$(word 4,$$(subst .., ,$$*)) \
         $$(word 1,$$(subst .., ,%)).$$(word 3,$$(subst .., ,$$*)).$$(word 4,$$(subst .., ,$$*))
	perl $(word 1,$^) $(word 2,$^) $(word 3,$^)  >  $@


