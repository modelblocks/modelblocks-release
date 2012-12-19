package edu.berkeley.nlp.syntax.binarization;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.ling.CollinsHeadFinder;
import edu.berkeley.nlp.syntax.Tree;


public abstract class HeadParentCommonBinarizer implements TreeBinarizer
{

	private static CollinsHeadFinder headFinder = new CollinsHeadFinder();

	/**
	 * Binarize a tree around its head, with the symbol names derived from
	 * either the parent or the head (as determined by binarization).
	 * <p>
	 * It calls {@link @headParentBinarizeTreeHelper} to do the messy work of
	 * binarization when that is actually necessary.
	 * 
	 * @param binarization
	 *            This determines whether the newly-created symbols are based on
	 *            the head symbol or on the parent symbol. It should be either
	 *            headBinarize or parentBinarize.
	 * @param tree
	 * @return
	 */
	public  Tree<String> binarizeTree(Tree<String> tree)
	{
		List<Tree<String>> children = tree.getChildren();
		if (children.size() == 0)
		{
			return tree;
		}
		else if (children.size() == 1)
		{
			List<Tree<String>> kids = new ArrayList<Tree<String>>(1);
			kids.add(binarizeTree(children.get(0)));
			return new Tree<String>(tree.getLabel(), kids);
		}
		else if (children.size() == 2)
		{
			List<Tree<String>> kids = new ArrayList<Tree<String>>(2);
			kids.add(binarizeTree(children.get(0)));
			kids.add(binarizeTree(children.get(1)));
			return new Tree<String>(tree.getLabel(), kids);
		}
		else
		{
			List<Tree<String>> kids = new ArrayList<Tree<String>>(1);
			kids.add(headParentBinarizeTreeHelper(tree, 0, children.size() - 1, headFinder.determineHead(tree), false, ""));
			return new Tree<String>(tree.getLabel(), kids);
		}
	}
	
	
	/**
	 * This method is the only way in which HEAD and PARENT binarization are different.
	 * @param tree
	 * @param head
	 * @return
	 */
	protected abstract String extractLabel(Tree<String> tree, Tree<String> head);

	/**
	 * This binarizes a tree into a bunch of binary [at]SYM-R symbols. It
	 * assumes that this sort of binarization is always necessary, so it is only
	 * called by {@link headParentBinarizeTree}.
	 * 
	 * @param binarization
	 *            The type of new symbols to generate, either head or parent.
	 * @param tree
	 * @param leftChild
	 *            The index of the leftmost child remaining to be binarized.
	 * @param rightChild
	 *            The index of the rightmost child remaining to be binarized.
	 * @param head
	 *            The head symbol of this level of the tree.
	 * @param right
	 *            This indicates whether we have gotten to the right of the head
	 *            child yet.
	 * @return
	 */
	private  Tree<String> headParentBinarizeTreeHelper(Tree<String> tree, int leftChild, int rightChild, Tree<String> head,
		boolean right, String productionHistory)
	{
		if (head == null) throw new Error("head is null");
		List<Tree<String>> children = tree.getChildren();

		// test if we've finally come to the head word
		if (!right && children.get(leftChild) == head) right = true;

		// prepare the parent label
		String label = extractLabel(tree,head);
		String parentLabel = "@" + label + (right ? "-R" : "-L") + "->" + productionHistory;

		// if the left child == the right child, then we only need a unary
		if (leftChild == rightChild)
		{
			ArrayList<Tree<String>> kids = new ArrayList<Tree<String>>(1);
			kids.add(binarizeTree(children.get(leftChild)));
			return new Tree<String>(parentLabel, kids);
		}

		// if we're to the left of the head word
		if (!right)
		{
			ArrayList<Tree<String>> kids = new ArrayList<Tree<String>>(2);
			Tree<String> child = children.get(leftChild);
			kids.add(binarizeTree(child));
			kids.add(headParentBinarizeTreeHelper(tree, leftChild + 1, rightChild, head, right, productionHistory + "_" + child.getLabel()));
			return new Tree<String>(parentLabel, kids);
		}
		// if we're to the right of the head word
		else
		{
			ArrayList<Tree<String>> kids = new ArrayList<Tree<String>>(2);
			Tree<String> child = children.get(rightChild);
			kids.add(headParentBinarizeTreeHelper( tree, leftChild, rightChild - 1, head, right, productionHistory + "_" + child.getLabel()));
			kids.add(binarizeTree(child));
			return new Tree<String>(parentLabel, kids);
		}
	}
}
