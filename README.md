README for ModelBlocks
======================

This is the `modelblocks` software package.  It includes several
resources for constructing and evaluating broad-coverage probabilistic
models of cognitive processes, organized into projects centered around
different tasks and data sets.

Quickstart
----------
To quickly get started using modelblocks, read through the quickstart
guide in the neighboring `QUICKSTART.md` file. Below is more info about
the structure of modelblocks.

Use of Makefiles
----------------
Each project exists in a subdirectory of the main modelblocks
directory.  In order to ensure the reproducibility of experiments
conducted using this resource, these project directories each contain
a Makefile, which specifies how data sets, output files, and
evaluation results are constructed.  Comments in these Makefiles
describe how these items are named.  Several working example items can
be constructed by typing `make all` in the relevant project directory.

Included Resources
------------------
Modelblocks makes use of several third-party data and software
resources.  Where licenses permit, these have been included directly
in the modelblocks package, so as to avoid version compatibility
issues and thereby ensure reproducibility.  In some cases open-source
software has been modified so as to produce a common data file format
required by other software.  All resources included in this package
are distributed under the Gnu General Public License (see LICENSE file
in this directory).

External Resources and `USER-*.TXT` Files
-----------------------------------------
When licenses of resources used in modelblocks projects do not permit
redistribution, or when (usually data) resources are too large to be
included, the Makefile will generate an appropriately-named
`user-*.txt` file, in which a user may specify a path to an external
copy of the resource (by default, modelblocks will assume a mounted
`/home/corpora/` directory at root, followed by the resource's name
and version number where relevant).  This use of `user-*.txt` files is
intended to allow users to specify external resources or other
user-specific data without having to modify the Makefile, which may be
overwritten in subsequent updates to modelblocks.
