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

THISDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# included...
#include $(dir $(CURDIR))general/Makefile
#include $(dir $(CURDIR))gcg/Makefile
#include $(dir $(CURDIR))lvpcfg/Makefile

SCRIPTS := $(THISDIR)scripts
INCLUDE := $(THISDIR)include
SRC     := $(THISDIR)src

BIN     := bin

include $(wildcard $(THISDIR)*/*.d)      ## don't comment out, it breaks make!

all: wsj23.wsj02to21-gcg14b-1671-3sm-bd.x-efabp.-c_-b2000_parsed.linetrees

################################################################################
#
#  i. Code compilation items
#
################################################################################

SHELL = /bin/bash
#INCLUDES = -I$(INCLUDE) -I../rvtl/include 
#CFLAGS = $(INCLUDES) -Wall `cat user-cflags.txt` -fpermissive
#CC = g++

#### obtain c++ dependencies by compiling local source and include files
.PRECIOUS: $(THISDIR)%.d
$(THISDIR)%.d:  $(THISDIR)%.cpp  $(INCLUDE)
	echo '$*.d: \' > $@   #' ##
	echo `g++ -I$(word 2,$^) -I../rvtl/include -MM $<` | sed 's/^.*:.*\.cpp */ $$(wildcard /;s/\\ *//g;s/$$/)/' >> $@
	cat $@ | sed 's/\.d:/\.cpp:/' >> $@
#       $(CC) -MM $< | sed 's/^.*://' >> $@

#### obtain binaries by compiling local source and include files
.PRECIOUS: bin/%
bin/%:  $(SRC)/$$(notdir $$@).cpp  $(SRC)/$$(notdir $$@).d  $(INCLUDE)  user-cflags.txt  |  bin  #$$(abspath $$(dir $$@))  #$$(realpath $$(dir $$@))  #$(BIN)  #$(CURDIR)/$$(notdir $$(dir $$@))  #
	g++ -I$(word 3,$^) -I../rvtl/include -Wall `cat $(word 4,$^)` -fpermissive -lm $< -o $@

################################################################################
#
#  1. Model building items
#
################################################################################

#### obtain '.model' format pcfg (with unaries) from berkeley text grammar
#.PRECIOUS: %.x-ccu.model
%.x-ccu.model:  %.grammar  %.lexicon  $(SCRIPTS)/berkgrammar2ckygr.py  $(SCRIPTS)/berklexicon2ckylex.py
	cat $(word 1,$^)  |  python3 $(word 3,$^)  >   $@
	cat $(word 2,$^)  |  python3 $(word 4,$^)  >>  $@

#### obtain strict binary-branching pcfg model (with no unaries) from pcfg model with unaries
#.PRECIOUS: %.x-cc.model
%.x-cc.model:  %.x-ccu.model  $(SCRIPTS)/ccu2cc.py
	cat $<  |  python3 $(word 2,$^)  >  $@

#### obtain branch- and depth-annotated grammar with preterm nodes generated seperately
#.PRECIOUS: %.bd.x-ccp.model
%.bd.x-ccp.model:  %.x-cc.model  $(SCRIPTS)/pcfg2pxmodel.py
	cat $<  |  python3 $(word 2,$^)  >  $@

#### obtain incremental probability model from strict binary-branching pcfg
#.PRECIOUS: %.x-fabp.probmodel
%.x-fabp.probmodel:  %.x-ccp.model  bin/calc-fabp-model
	cat $<  |  grep -v '^PX '  | sed 's/^Cr : \(.*\^[Ll],1\) =/CC REST^R,0 : \1 REST^R,0 =/;s/ - - / -\^.,. -\^.,. /' \
                |  sed 's/CC \(.*\)\^\(.\),\(.\) : \(.*\)\^.,. \(.*\)\^.,. = \(.*\)/CC \2 \3 \1 : \4 \5 = \6/' |  $(word 2,$^)  >  $@

#### obtain sorted model for A* algo, and rename for parser that incrementally tracks entities
.PRECIOUS: %.x-efabp.model
%.x-efabp.model:  %.x-fabp.probmodel  $(SCRIPTS)/sortbyprob.pl
	cat $<  |  perl $(word 2,$^)  >  $@

################################################################################
#
#  2. Parsing items
#
################################################################################

##### obtain model-specific parser output by running sentences through parser given flags and model, from including makefile's genmodel dir:
##### <testset>.<trainset>.<model>.(<params>_)parsed  ---->  genmodel/<testset>.sents  bin/parser-<model>  genmodel/<trainset>.<model>.model
# $$(basename $$(basename $$(basename %))).$$(findstring hy,$$*)sents
.PRECIOUS: %parsed.output
%parsed.output:  $$(basename $$(basename $$(basename %))).sents \
                 $(BIN)/parser-$$(subst .,,$$(suffix $$(basename $$*))) \
                 $(GENMODEL)/$$(subst -,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))$$(suffix $$(basename $$*)).model
	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
	@sleep 5
	cat $<  |  $(word 2,$^)  $(subst _, ,$(subst .,,$(suffix $*)))  $(word 3,$^)  >  $@

#### obtain linetrees by converting output using script:
#%parsed.linetrees: $$(word 1,$$(subst ., ,%))$$(suffix $$(basename $$*))$$(suffix $$*)parsed.output \
#                   $(SCRIPTS)/$$(lastword $$(subst -, ,$$(basename $$*)))out2linetrees.py \
#                   $(SCRIPTS)/remove-at-cats.pl
%parsed.linetrees: %parsed.output $(SCRIPTS)/$$(lastword $$(subst -, ,$$(basename $$*)))out2linetrees.py  $(SCRIPTS)/remove-at-cats.pl
	#both sed calls are necessary since the patterns overlap, so two passes with sed are necessary to convert all plusops to dashops
	cat $<  |  python3 $(word 2,$^)  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  |  perl $(word 3,$^)  >  $@

