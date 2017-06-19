QUICK START
===========

Basic File Structure
--------------------
Modelblocks is composed of RESOURCE and PROJECT subdirectories.
RESOURCE subdirectories build resources that are used by a variety of
projects (e.g., the GCG-reannotated WSJ corpus or the left-corner
parser). PROJECTs contain individual projects. By convention,
all RESOURCE directory names begin with the string "resource-".
PROJECT directories with recipes to reproduce published results
are named by author/year. ModelBlocks also provides a universal
PROJECT directory (called `workspace`) for sandboxing and development.

Initializing and using the workspace
------------------
When you first set up your repository, you should navigate to the
repository root and type `make`. This will set up a new subdirectory
called `workspace` from which you can build nearly every available
ModelBlocks recipe.

If you are an experiment developer, you can use the Makefile contained
in the workspace as a template for building your own project directory.
The workspace Makefile provides includes to all relevant internal and
external resources used by ModelBlocks. However, for external resources
(e.g. treebanks, experimental data, etc.) you will need to acquire the
resource and update the relevant user-*.txt pointer file in the `config`
directory of ModelBlocks before you can make any recipes that depend on
it.

There is no danger in making a target that has a missing dependency.
Make will just exit with a description of the missing resource and how
to access it. Therefore, a reasonable way of discovering all the
dependencies of a given target is to repeatedly make it and follow
the error reports until all dependencies are satisfied.

Example
------------------------
Let's imagine you'd like to use the left-corner incremental parser
provided by ModelBlocks to get incremental complexity metrics for 
section 23 of the Wall Street Journal corpus, training on sections
2 through 21. The target you will need to make (from the workspace
directory) is:

genmodel/wsj23.wsj02to21-nodashtags-5sm-bd-x+efabp-+c_+b5000.pcfg.tokmeasures

(See the docs and/or ModelBlocks Assistant for further explanation
of the components of this target)

The only external resources necessary to make this target are

- Penn Treebank 3
- Berkeley Parser jarfile (for grammar training)

For details on how to access these resources, make the target and
read the error reports. If you already have them on your system,
make the target in order to initialize the config/user-*.txt pointer
files, then update config/user-treebank-directory.txt and
config/user-berkeleyparserjar-directory.txt to point to the correct
locations.

Once your dependencies are satisfied, the target should successfully
execute. It may take several hours to complete, but once finished,
the target file should contain word-by-word complexity metrics for
WSJ 23.

Getting help
------------------------
ModelBlocks automates a large array of data processing and analysis
routines, text corpora, and experiment datasets. It can therefore
be overwhelming to figure out where to begin. There are three ways
to get help using ModelBlocks:

1. Contact us (the developers) through Github: You can reach out privately
by email to anyone listed as a contributor to the project, and/or
you can publicly post an issue to the Issues page. We are happy to
help anyone interested in using this resource, no matter how basic
the question may seem.

2. Explore using ModelBlocks Assistant: ModelBlocks Assistant is an 
interactive browser-based target builder that supports the most commonly-
used ModelBlocks targets. ModelBlocks Assistant provides a GUI that
allows you to make choices about the various component modules of a
target and automatically generates the corresponding well-formed
Make command. This is a good way to quickly get the format of a target
you want to build, as well as to play around with the components of a
target in order to get a better handle on the syntax.

ModelBlocks Assistant can be accessed remotely at http://go.osu.edu/mbassist
or built locally by executing `make mbassist` from the `doc` directory.

Targets should generally be made from the `workspace` directory.
ModelBlocks Assistant is still in beta. If you run into any issues
using it, we'd very much appreciate hearing from you.

3. Read the docs: Navigate to the `doc` repository and run
`make tech-report.pdf` for the most complete documentation available.
Our docs are constantly being updated as ModelBlocks evolves, so if
you find something that you believe is incorrect or out of date,
please feel free to contact us.

