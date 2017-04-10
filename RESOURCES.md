MODELBLOCKS RESOURCES
===========

This file describes the purpose of each resource-* directory, so users will have a better idea of when each should be included. Any directory with [*] links to resources that are not distributed with Modelblocks and so must be obtained separately.

resource-bmmm
--------------------
TBA

resource-bnc[*]
------------------
This directory provides hooks to the British National Corpus (BNC; BNC Consortium 2007).

resource-ccl
-----------------------
TBA

resource-childes[*]
--------------------------
Provides hooks for the Adam, Eve, and Sarah portions of the CHILDES corpus (Macwhinney 2000).

resource-dmv
------------------------
TBA

resource-dundee[*]
------------------------
Provides hooks for the Dundee eye-tracking corpus (Kennedy et al., 2003).

resource-gcg
------------------------
Provides tools for the Nguyen et al. (2012) Generalized Categorial Grammar (GCG).

resource-general
------------------------
TBA

resource-gigaword[*]
------------------------
Provides hooks for the Gigaword text corpus (Graff and Cieri, 2003).

resource-incrsem
------------------------
TBA

resource-kenlm[*]
------------------------
Provides hooks to the KenLM N-gram Language Modeling toolkit (Heafield et al., 2013).

resource-lcparse
------------------------
Provides hooks to the van Schijndel et al., (2013) left-corner parser, which computes a variety of incremental complexity estimates.

resource-linetrees
------------------------
Provides tools to convert sentences to the expected linetrees format as well as running parser evaluations via evalb. Needed for parsing.

resource-lmefit
------------------------
Provides tools for mixed effects models using lmer (Bates et al., 2014). In particular, this directory provides diamond ANOVA recipes for computing significance via likelihood-ratio testing.

resource-logreg-torch
------------------------
TBA

resource-logreg
------------------------
TBA

resource-lvpcfg
------------------------
Provides hooks to the Petrov and Klein (2007) parser (aka the Berkeley Parser), and the associated split-merge grammar trainer (Petrov et al., 2006).

resource-naturalstories[*]
------------------------
Provides hooks to the Natural Stories self-paced reading corpus (Futrell et al., in prep).

resource-ontonotes[*]
------------------------
Provides hooks to the Ontonotes corpus (Weischedel et al., 2013).

resource-rhacks[*]
------------------------
Provides hooks to externally available R tools, especially for mixed effects models.

resource-rt
------------------------
Provides tools for conducting corpus reading time experiments. For example, accumulating predictor measures, computing DLT, coreference retrieval, etc. (Perhaps this should be renamed resource-behavioral?)

resource-rvtl
------------------------
Random variable template library for C++. Used by several C++ components of Modelblocks. Not typically a required include for project directories.

resource-segment-tokenize
------------------------
TBA

resource-srilm[*]
------------------------
Provides hooks to the SRI N-gram Language Modeling toolkit (SRILM; Stolcke, 2002). Generally deprecated in favor of KenLM now.

resource-tiger[*]
------------------------
Provides hooks to the Tiger German treebank corpus (Brants et al., 2004).

resource-tokenizer[*]
------------------------
Provides hook to external extended Penn tokenizer, which tries to tokenize sentences as in the Penn Treebank to standardize input to parsers trained on that data.

resource-treebank[*]
------------------------
Provides hooks to the Penn Treebank corpus (Marcus et al., 1993). Often used to train the parser.

resource-ucl[*]
------------------------
Provides hooks to the University College London (UCL) eye-tracking corpus (Frank et al., 2013).

resource-upparse
------------------------
TBA

resource-wordnet[*]
------------------------
`BROKEN` Provides hooks to the wordnet corpus. *Currently missing the wordnet2hyps.py script.*

References
==========

Douglas Bates, Martin Maechler, Ben Bolker, and Steven Walker. lme4: Linear mixed-
effects models using Eigen and S4, 2014. URL http://CRAN.R-project.org/
package=lme4. R package version 1.1-7.

The British National Corpus, version 3 (BNC XML Edition). 2007. Distributed by Oxford University Computing Services on behalf of the BNC Consortium. URL: http://www.natcorp.ox.ac.uk/ 

Sabine Brants, Stefanie Dipper, Peter Eisenberg, Silvia Hansen, Esther König, Wolfgang Lezius, Christian Rohrer, George Smith, and Hans Uszkoreit. 2004. TIGER: Linguistic Interpretation of a German Corpus. Journal of Language and Computation, 2004 (2), 597-620.

Stefan L. Frank, Irene Fernandez Monsalve, Robin L. Thompson, and Gabriella Vigliocco. 2013. Reading time data for evaluating broad-coverage models of English sentence processing. Behavior Research
Methods, 45:1182–1190.

Richard Futrell, Edward Gibson, Hal Tily, Anastasia Vishnevetsky, Steve Piantadosi, and
Evelina Fedorenko. Natural stories corpus. in prep.

David Graff and Christopher Cieri. English Gigaword LDC2003T05, 2003.

Kenneth Heafield, Ivan Pouzyrevsky, Jonathan H. Clark, and Philipp Koehn. Scalable modified Kneser-Ney language model estimation. In Proceedings of the 51st Annual Meeting
of the Association for Computational Linguistics, pages 690–696, Sofia, Bulgaria, August 2013.

Alan Kennedy, James Pynte, and Robin Hill. The Dundee corpus. In Proceedings of the
12th European conference on eye movement, 2003.

Brian MacWhinney. (2000). The CHILDES Project: Tools for analyzing talk. Third Edition. Mahwah, NJ: Lawrence Erlbaum Associates.

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large
annotated corpus of English: the Penn Treebank. Computational Linguistics, 19(2):
313–330, 1993.

Luan Nguyen, Marten van Schijndel, and William Schuler. Accurate unbounded depen-
dency recovery using generalized categorial grammars. In Proceedings of the 24th In-
ternational Conference on Computational Linguistics (COLING ’12), pages 2125–2140,
Mumbai, India, 2012.

Slav Petrov and Dan Klein. Improved inference for unlexicalized parsing. In Proceedings
of NAACL HLT 2007, pages 404–411, Rochester, New York, April 2007. Association
for Computational Linguistics.

Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact,
and interpretable tree annotation. In Proceedings of the 44th Annual Meeting of the
Association for Computational Linguistics (COLING/ACL’06), 2006.

Andreas Stolcke. 2002. SRILM - an extensible language modeling toolkit. In
Proceedings of the Seventh International Conference on Spoken Language Processing, pages 901–904.

Marten van Schijndel, Andy Exley, and William Schuler. A model of language processing
as hierarchic sequential prediction. Topics in Cognitive Science, 5(3):522–540, 2013.

Ralph Weischedel, Martha Palmer, Mitchell Marcus, Eduard Hovy, Sameer Pradhan, Lance Ramshaw, Nianwen Xue, Ann Taylor, Jeff Kaufman, Michelle Franchini, Mohammed El-Bachouti, Robert Belvin, Ann Houston. Ontonotes v5.0. 2013.