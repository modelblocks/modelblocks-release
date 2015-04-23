package edu.berkeley.nlp.syntax.binarization;

import edu.berkeley.nlp.syntax.Tree;


/**
 * Binarize a tree around the head symbol. That is, when there is an n-ary
 * rule, with n > 2, we split it into a series of binary rules with titles
 * like [AT]JJ-R (if JJ is the head of the rule). The right part of the
 * symbol (-R or -L) is used to indicate whether we're producing to the
 * right or to the left of the head symbol. Thus, the head symbol is always
 * the deepest symbol on the tree we've created.
 * 
 *
 */
public class ParentBinarizer extends HeadParentCommonBinarizer
{

	@Override
	protected String extractLabel(Tree<String> tree, Tree<String> head)
	{
		return tree.getLabel();
	}

}
