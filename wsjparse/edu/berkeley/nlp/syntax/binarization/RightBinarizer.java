package edu.berkeley.nlp.syntax.binarization;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;

public class RightBinarizer implements TreeBinarizer
{
	public Tree<String> binarizeTree(Tree<String> tree)
	{
		String label = tree.getLabel();
		List<Tree<String>> children = tree.getChildren();
		if (tree.isLeaf())
			return tree.shallowCloneJustRoot();
		else if (children.size() == 1) { return new Tree<String>(label, Collections.singletonList(binarizeTree(children.get(0)))); }
		// otherwise, it's a binary-or-more local tree, so decompose it into a
		// sequence of binary and unary trees.
		String intermediateLabel = "@" + label + "->";
		Tree<String> intermediateTree = rightBinarizeTreeHelper(tree, children.size() - 1, intermediateLabel);
		return new Tree<String>(label, intermediateTree.getChildren());
	}
	
	

	private  Tree<String> rightBinarizeTreeHelper(Tree<String> tree, int numChildrenLeft, String intermediateLabel)
	{
		Tree<String> rightTree = tree.getChildren().get(numChildrenLeft);
		List<Tree<String>> children = new ArrayList<Tree<String>>(2);
		if (numChildrenLeft == 1)
		{
			children.add(binarizeTree(tree.getChildren().get(numChildrenLeft - 1)));
		}
		else if (numChildrenLeft > 1)
		{
			Tree<String> leftTree = rightBinarizeTreeHelper(tree, numChildrenLeft - 1, intermediateLabel + "_" + rightTree.getLabel());
			children.add(leftTree);
		}
		children.add(binarizeTree(rightTree));
		return new Tree<String>(intermediateLabel, children);
	}

}
