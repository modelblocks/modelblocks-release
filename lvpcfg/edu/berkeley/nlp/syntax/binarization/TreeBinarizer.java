package edu.berkeley.nlp.syntax.binarization;

import edu.berkeley.nlp.syntax.Tree;

public interface TreeBinarizer
{
	Tree<String> binarizeTree(Tree<String> tree);
}
