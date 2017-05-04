
SHELL = /bin/bash

LATEX=pdflatex
DVIPS=dvips
PS2PDF=ps2pdf
BIBTEX=bibtex

LATEX_OPTS=-interaction=nonstopmode -halt-on-error
DVIPS_OPTS=-Ppdf -t letter
PS2PDF_OPTS=-dPDFSETTINGS=/prepress -dSubsetFonts=true -dEmbedAllFonts=true
NUM=""

RERUN = "(There were undefined references|Rerun to get (cross-references|the bars) right)"
RERUNBIB = "No file.*\.bbl|Citation.*undefined"


.PHONY: all clean


all: paper_draft.dvi paper_draft.bbl

pdf: paper_draft.pdf

%: %.pdf *.tex

%.pdf : %.ps
	$(PS2PDF) $(PS2PDF_OPTS) $<

%.ps: %.dvi
	$(DVIPS) $(DVIPS_OPTS) $<

%.pdf: %.tex $(wildcard *.tex) $(wildcard ../styles/*.sty)
	$(LATEX) $(LATEX_OPTS) $<
	$(BIBTEX) $(patsubst %.tex,%,$<)
	$(LATEX) $(LATEX_OPTS) $<
	$(LATEX) $(LATEX_OPTS) $<
	rm -f $(patsubst %.tex,%,$<).log

%.bbl: %.tex ../../bibliography/bibliography_luan.bib
	if [ -n %.aux ]; \
	then \
		$(LATEX) $(LATEX_OPTS) $<;\
	fi
	$(BIBTEX) $(patsubst %.tex,%,$<)

%.again: %.tex
	$(LATEX) $(LATEX_OPTS) $<

### \rm *.typos
### foreach i ( paper_section_*.tex )
### make $i:r.typos
### end
### more *.typos
%.typos: %.tex
	cat $^ | sed 's/%.*//g;s/\\usepackage{[^}]*}//g;s/\\bibliographystyle{[^}]*}//g;s/\\cite{[^}]*}//g;s/\\label{[^}]*}//g;s/\\ref{[^}]*}//g;s/\\input{[^}]*}//g;s/\\begin{[^}]*}//g;s/\\end{[^}]*}//g;s/\\psset{[^}]*}//g;s/\\[a-zA-Z]*//g;s/\[.*\]//g;s/\$$[^$$]*\$$//g' | spell | sort -u > $@


clean:
	+rm -fv *.{dvi,ps,aux,log,bbl,blg,synctex.gz} *~ \#*

########################################
#
#  ModelBlocks Assistant
#
########################################

yaml2json:
	python generateJsonData.py
