QUICK START
===========

Basic File Structure
--------------------
Modelblocks is composed of `HUB` and `TERMINAL` subdirectories.
`HUB` subdirectories build resources that are used by a variety of
projects (e.g., the GCG-reannotated WSJ corpus or the left-corner
parser). `TERMINALS` contain individual projects.

To begin, you'll need a `TERMINAL` subdirectory devoted to your project
which is appropriately linked to the `HUB` directories. To easily
accomplish this, copy `vanschijndelschuler15` and rename the result
for your project. For this quickstart, we'll call your project
`newproj`.

    > cp -r vanschijndelschuler15 newproj
    > cd newproj

Test Corpus Construction
------------------------
Most likely, you'll want to extract incremental complexity metrics
for some corpus (referred to as `NEWCORPUS`). You'll need to get your
corpus into `newproj/genmodel/NEWCORPUS.sents`. The sents file needs
to have one sentence per line and be tokenized comparably to the
data used to train the complexity metrics. This guide will assume
that you will use the Penn Treebank for training. In that case,
`NEWCORPUS.sents` will need to use the PTB tokenization scheme.
You can get a PTB tokenizer from:
https://github.com/vansky/extended_penn_tokenizer

to use the tokenizer:
    > extended_penn_tokenizer/ptb_tokenizer.sed < NEWCORPUS.untoksents > NEWCORPUS.sents

Training Complexity Metrics
---------------------------
Modelblocks needs to be told where to access tools that don't come
bundled with it, such as the Penn Treebank.

    > make prereqs

The above command will make a variety of `user-*.txt` files, which
each point to a given external resource. For example,
`user-treebank-directory.txt` points to the Wall Street Journal
corpus. You need to edit these files to point to your local
locations of each resource. You won't need all the resources,
though. Mainly, you'll just need the WSJ for parsing and incremental
complexity metrics. The other `user-*.txt` files are for complete
replication of van Schijndel and Schuler (2015). Aside from reading
through the Makefile, you can tell if you need a given dependency by
trying to make your desired target and having make complain about a
missing dependency. Since these dependencies are external to modelblocks,
you'll need to have them installed independently and will need to refer
to their documentation for using them (e.g., for generating your own
kenlm n-gram models). You can refer to the specific Makefile item that
uses a given `user-*.txt` file to see how the resource is used (just search
for that `user-*.txt` within the Makefile).

Got everything redirected, correctly? Okay! Then you should
be able to get complexity metrics. Computing the complexity
metrics leads to a fairly slow parser. Enter a Screen first,
so you can come back later when it's done.

    > screen
    > make NEWCORPUS.wsj02to21-gcg14-1671-3sm-bd.x-efabp.-c_-b5000.complextoks

To leave the screen:
    > Ctrl+a d

To reaccess the screen to check on the build progress:
    > screen -r

You can look at the output parses in:
`NEWCORPUS.wsj02to21-gcg14-1671-3sm-bd.x-efabp.-c_-b5000_parsed.output`

The complexity metrics will be output to the `%.complextoks`
file. The first line is a header. Each other line begins with
the next word in the corpus followed by the incremental
complexity metrics for that point in the sentence.

Filename Explanation
--------------------
The following is a breakdown of the filenames that make 'parses'
to generate output files:

`wsj02to21-gcg14` refers to the training corpus, which consists of 
Sections 02 to 21 of the WSJ reannotated with the Nguyen et al. 
(2012) grammar tags (`genmodel/wsj02to21.gcg14.linetrees`)

`1671` is the standard portion of the training set used to tune
the split-merge tagset (the last 1671 lines)

`5sm` refers to the number of split-merge iterations. Five 
was shown to be optimal without overfitting by Petrov et al. (2007).

`bd` means the grammar is side- and depth-specific (parser juju).

`x-efabp` is the parser

`-c` is a parser flag that tells it to output complexity metrics

`-b2000` is a parser flag that tells it to run with a beam-width of 2000.
2000 was shown to be optimal for accuracy with the 5sm PTB tagset by 
van Schijndel et al (2013). We've recently found that larger beams 
improve the fit to reading times even though the parse accuracy remains 
the same.

For reading time fits, we recommend a beam-width of 5000, but note
that this will greatly slow the parser, so maybe pilot at 2000

Complexity Output
-----------------
 * `totsurp`: Total suprisal (Hale 2001)
 * `lexsurp`: Lexical surprisal (Roark 2009)
 * `synsurp`: Syntactic surprisal (Roark 2009)

Those are the big guns. They are highly correlated with a variety of
cognitive measures including reading times, some ERPs, reaction times,
etc. Total surprisal is the actual measure, but Roark provided a
principled way of splitting it into the amount of surprisal
contributed by the lexical item and the amount of surprisal
contributed by the syntactic structure (they sum to totsurp). The
cognitive explanation behind surprisal is that it's the amount
of reshuffling of possibilities that your brain has to do.
Essentially, it's the amount of probability mass you assigned
to parse hypotheses that have now been ruled out (or at least
changed in probability).

 * `entred`: Entropy reduction (Hale 2006)

Entropy reduction is the amount of uncertainty that has been ruled out
or introduced (negative reduction) by the current observation. Whereas
surprisal predicts the amount of probability mass that had to be
reallocated, entropy reduction deals with changes in uncertainty.

 * `embdep`: Embedding depth (Chomsky & Miller 1963?)

This measure has been around forever and likely predates the above
citation. It's just a measure of how many unresolved (center
embedding) dependencies are currently in working memory. Essentially,
it's a proxy for memory load.

 * `embdif`: Embedding difference (Wu et al., 2010)

Kind of similar to surprisal but applied to memory load. Essentially,
it measures the number of changes in center embedding that occurred
over all of the possible parse hypotheses. If you believe in parallel
parsing, it measures the weighted number of superposed memory
operations you'd need to conduct to continue parsing. When used
over a sentence or a complete dependency arc, it basically tells
you how many garden paths were possible in that span (otherwise
it will be 1).

The rest of the measures are specific syntactic parser operations. If
you're interested in these, take a look at:
van Schijndel and Schuler (2013)
van Schijndel, Nguyen and Schuler (2013)
or contact us for more info.

The final column is:

* `sentid`: Sentence ID

Importantly, all of the measures are computed over the 'entire' (top
5000 options) hypothesis space, *not* just from the best parse. This
means, for example, that the embedding depth is often a real number as
each parse, weighted by its probability, changes embedding depths.

Good luck! Let us know if you encounter bugs or have questions.

These were adapted from the online instructions:
http://www.ling.ohio-state.edu/~vanschm/modelblocks_instructs

For more info, check out the FAQ:
http://www.ling.ohio-state.edu/~vanschm/modelblocks_help
