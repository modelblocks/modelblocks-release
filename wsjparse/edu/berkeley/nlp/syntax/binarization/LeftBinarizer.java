package edu.berkeley.nlp.syntax.binarization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;

public class LeftBinarizer implements TreeBinarizer
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
		Tree<String> intermediateTree = leftBinarizeTreeHelper(tree, 0, intermediateLabel);
		return new Tree<String>(label, intermediateTree.getChildren());
	}
	

	

	private  Tree<String> leftBinarizeTreeHelper(Tree<String> tree, int numChildrenGenerated, String intermediateLabel)
	{
		Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
		List<Tree<String>> children = new ArrayList<Tree<String>>(2);
		children.add(binarizeTree(leftTree));
		if (numChildrenGenerated == tree.getChildren().size() - 2)
		{
			children.add(binarizeTree(tree.getChildren().get(numChildrenGenerated + 1)));
		}
		else if (numChildrenGenerated < tree.getChildren().size() - 2)
		{
			Tree<String> rightTree = leftBinarizeTreeHelper(tree, numChildrenGenerated + 1, intermediateLabel + "_" + leftTree.getLabel());
			children.add(rightTree);
		}
		return new Tree<String>(intermediateLabel, children);
	}


}
