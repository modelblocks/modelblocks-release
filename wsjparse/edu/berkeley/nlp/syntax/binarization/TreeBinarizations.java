package edu.berkeley.nlp.syntax.binarization;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import edu.berkeley.nlp.util.Filter;

/**
 * Class which contains code for annotating and binarizing trees for the
 * parser's use, and debinarizing and unannotating them for scoring.
 */
public class TreeBinarizations {

	/**
	 * This annotates the parse tree by adding ancestors to the tags, and then
	 * by forgetfully binarizing the tree. The format goes as follows: Tag
	 * becomes Tag^Parent^Grandparent Then, this is binarized, so that
	 * Tag^Parent^Grandparent produces A^Tag^Parent B... C... becomes
	 * Tag^Parent^Grandparent produces A^Tag^Parent
	 * 
	 * @Tag^Parent^Grandparent-&gt;_A^Tag^Parent
	 * @Tag^Parent^Grandparent-&gt;_A^Tag^Parent produces B^Tag^Parent
	 * @Tag^Parent^Grandparent-&gt;_A^Tag^Parent_B^Tag^Parent and finally we
	 *                                                        trim the excess _*
	 *                                                        off to control the
	 *                                                        amount of
	 *                                                        horizontal history
	 * 
	 */
	public static Tree<String> processTree(Tree<String> unAnnotatedTree,
			int nVerticalAnnotations, int nHorizontalAnnotations, TreeBinarizer binarization,
			boolean manualAnnotation) {
		Tree<String> verticallyAnnotated = (nVerticalAnnotations == 0) ? unAnnotatedTree
				: verticallyAnnotate(unAnnotatedTree, nVerticalAnnotations, manualAnnotation);
		Tree<String> binarizedTree = binarization.binarizeTree(verticallyAnnotated);
		// removeUnaryChains(binarizedTree);
		// System.out.println(binarizedTree);

		// if (deleteLabels) return deleteLabels(binarizedTree,true);
		// else if (deletePC) return deletePC(binarizedTree,true);
		// else
		return nHorizontalAnnotations == Integer.MAX_VALUE ? binarizedTree : horizontallyMarkovize(binarizedTree, nHorizontalAnnotations);

	}

	private static Tree<String> verticallyAnnotate(Tree<String> unAnnotatedTree,
			int nVerticalAnnotations, boolean manualAnnotation) {
		Tree<String> verticallyAnnotated;
		if (nVerticalAnnotations == 3) {
			verticallyAnnotated = annotateVerticallyTwice(unAnnotatedTree, "", "");
		} else if (nVerticalAnnotations == 2) {
			if (manualAnnotation) {
				verticallyAnnotated = annotateManuallyVertically(unAnnotatedTree, "");
			} else {
				verticallyAnnotated = annotateVertically(unAnnotatedTree, "");
			}
		} else if (nVerticalAnnotations == 1) {
			verticallyAnnotated = markGrammarNonterminals(unAnnotatedTree, "");
		} else {
			throw new UnsupportedOperationException(
					"the code does not exist to annotate vertically " + nVerticalAnnotations
							+ " times");
		}
		return verticallyAnnotated;
	}

	private static Tree<String> annotateVerticallyTwice(Tree<String> tree,
			String parentLabel1, String parentLabel2) {
		Tree<String> verticallyMarkovizatedTree;
		if (tree.isLeaf()) {
			verticallyMarkovizatedTree = tree; // new
			// Tree<String>(tree.getLabel());//
			// + parentLabel);
		} else {
			List<Tree<String>> children = new ArrayList<Tree<String>>(tree.getChildren().size());
			for (Tree<String> child : tree.getChildren()) {
				// children.add(annotateVerticallyTwice(child,
				// parentLabel2,"^"+tree.getLabel()));
				children.add(annotateVerticallyTwice(child, "^" + tree.getLabel(), parentLabel1));
			}
			verticallyMarkovizatedTree = new Tree<String>(tree.getLabel() + parentLabel1
					+ parentLabel2, children);
		}
		return verticallyMarkovizatedTree;
	}

	private static Tree<String> annotateVertically(Tree<String> tree, String parentLabel) {
		Tree<String> verticallyMarkovizatedTree;
		if (tree.isLeaf()) {
			verticallyMarkovizatedTree = tree;// new
			// Tree<String>(tree.getLabel());//
			// + parentLabel);
		} else {
			List<Tree<String>> children = new ArrayList<Tree<String>>(tree.getChildren().size());
			for (Tree<String> child : tree.getChildren()) {
				children.add(annotateVertically(child, "^" + tree.getLabel()));
			}
			verticallyMarkovizatedTree = new Tree<String>(tree.getLabel() + parentLabel,
					children);
		}
		return verticallyMarkovizatedTree;
	}

	private static Tree<String> markGrammarNonterminals(Tree<String> tree,
			String parentLabel) {
		Tree<String> verticallyMarkovizatedTree;
		if (tree.isPreTerminal()) {
			verticallyMarkovizatedTree = tree;// new
			// Tree<String>(tree.getLabel());//
			// + parentLabel);
		} else {
			List<Tree<String>> children = new ArrayList<Tree<String>>(tree.getChildren().size());
			for (Tree<String> child : tree.getChildren()) {
				children.add(markGrammarNonterminals(child, "^g"));//
			}
			verticallyMarkovizatedTree = new Tree<String>(tree.getLabel() + parentLabel,
					children);
		}
		return verticallyMarkovizatedTree;
	}

	private static Tree<String> annotateManuallyVertically(Tree<String> tree,
			String parentLabel) {
		Tree<String> verticallyMarkovizatedTree;
		if (tree.isPreTerminal()) {
			// split only some of the POS tags
			// DT, RB, IN, AUX, CC, %
			String label = tree.getLabel();
			if (label.contains("DT") || label.contains("RB") || label.contains("IN")
					|| label.contains("AUX") || label.contains("CC") || label.contains("%")) {
				verticallyMarkovizatedTree = new Tree<String>(tree.getLabel() + parentLabel, tree
						.getChildren());
			} else {
				verticallyMarkovizatedTree = tree;// new
				// Tree<String>(tree.getLabel());//
				// + parentLabel);
			}
		} else {
			List<Tree<String>> children = new ArrayList<Tree<String>>(tree.getChildren().size());
			for (Tree<String> child : tree.getChildren()) {
				children.add(annotateManuallyVertically(child, "^" + tree.getLabel()));
			}
			verticallyMarkovizatedTree = new Tree<String>(tree.getLabel() + parentLabel,
					children);
		}
		return verticallyMarkovizatedTree;
	}

	// replaces labels with three types of labels:
	// X, @X=Y and Z
	private static Tree<String> deleteLabels(Tree<String> tree, boolean isRoot) {
		String label = tree.getLabel();
		String newLabel = "";
		if (isRoot) {
			newLabel = label;
		} else if (tree.isPreTerminal()) {
			newLabel = "Z";
			return new Tree<String>(newLabel, tree.getChildren());
		} else if (label.charAt(0) == '@') {
			newLabel = "@X";
		} else
			newLabel = "X";

		List<Tree<String>> transformedChildren = new ArrayList<Tree<String>>(tree.getChildren().size());
		for (Tree<String> child : tree.getChildren()) {
			transformedChildren.add(deleteLabels(child, false));
		}
		return new Tree<String>(newLabel, transformedChildren);
	}

	// replaces phrasal categories with
	// X, @X=Y but keeps POS-tags
	private static Tree<String> deletePC(Tree<String> tree, boolean isRoot) {
		String label = tree.getLabel();
		String newLabel = "";
		if (isRoot) {
			newLabel = label;
		} else if (tree.isPreTerminal()) {
			return tree;
		} else if (label.charAt(0) == '@') {
			newLabel = "@X";
		} else
			newLabel = "X";

		List<Tree<String>> transformedChildren = new ArrayList<Tree<String>>(tree.getChildren().size());
		for (Tree<String> child : tree.getChildren()) {
			transformedChildren.add(deletePC(child, false));
		}
		return new Tree<String>(newLabel, transformedChildren);
	}

	private static Tree<String> horizontallyMarkovize(Tree<String> tree,
			int nHorizontalAnnotation) {
		String transformedLabel = tree.getLabel();
		if (tree.isLeaf()) {
			return tree.shallowCloneJustRoot();
		}
		// the location of the farthest _
		int firstCutIndex = transformedLabel.indexOf('_');
		int keepBeginning = firstCutIndex;
		// will become -1 when the end of the line is reached
		int secondCutIndex = transformedLabel.indexOf('_', firstCutIndex + 1);
		// the location of the second farthest _
		int cutIndex = secondCutIndex;
		while (secondCutIndex != -1) {
			cutIndex = firstCutIndex;
			firstCutIndex = secondCutIndex;
			secondCutIndex = transformedLabel.indexOf('_', firstCutIndex + 1);
		}
		if (nHorizontalAnnotation == 0) {
			cutIndex = transformedLabel.indexOf('-');
			if (cutIndex > 0) transformedLabel = transformedLabel.substring(0, cutIndex);
		} else if (cutIndex > 0 && !tree.isLeaf()) {
			if (nHorizontalAnnotation == 2) {
				transformedLabel = transformedLabel.substring(0, keepBeginning)
						+ transformedLabel.substring(cutIndex);
			} else if (nHorizontalAnnotation == 1) {
				transformedLabel = transformedLabel.substring(0, keepBeginning)
						+ transformedLabel.substring(firstCutIndex);
			} else {
				throw new UnsupportedOperationException(
						"code does not exist to horizontally annotate at level "
								+ nHorizontalAnnotation);
			}
		}
		List<Tree<String>> transformedChildren = new ArrayList<Tree<String>>(tree.getChildren().size());
		for (Tree<String> child : tree.getChildren()) {
			transformedChildren.add(horizontallyMarkovize(child, nHorizontalAnnotation));
		}
		/*
		 * if (!transformedLabel.equals("ROOT")&& transformedLabel.length()>1){
		 * transformedLabel = transformedLabel.substring(0,2); }
		 */

		/*
		 * if (tree.isPreTerminal() && transformedLabel.length()>1){ if
		 * (transformedLabel.substring(0,2).equals("NN")){ transformedLabel =
		 * "NNX"; } else if (transformedLabel.equals("VBZ") ||
		 * transformedLabel.equals("VBP") || transformedLabel.equals("VBD") ||
		 * transformedLabel.equals("VB") ){ transformedLabel = "VBX"; } else if
		 * (transformedLabel.substring(0,3).equals("PRP")){ transformedLabel =
		 * "PRPX"; } else if (transformedLabel.equals("JJR") ||
		 * transformedLabel.equals("JJS") ){ transformedLabel = "JJX"; } else if
		 * (transformedLabel.equals("RBR") || transformedLabel.equals("RBS") ){
		 * transformedLabel = "RBX"; } else if (transformedLabel.equals("WDT") ||
		 * transformedLabel.equals("WP") || transformedLabel.equals("WP$")){
		 * transformedLabel = "WBX"; } }
		 */
		return new Tree<String>(transformedLabel, transformedChildren);
	}

	public static Tree<String> unAnnotateTreeSpecial(Tree<String> annotatedTree) {
		// Remove intermediate nodes (labels beginning with "Y"
		// Remove all material on node labels which follow their base symbol
		// (cuts at the leftmost -, ^, or : character)
		// Examples: a node with label @NP->DT_JJ will be spliced out, and a
		// node with label NP^S will be reduced to NP
		Tree<String> debinarizedTree = Trees.spliceNodes(annotatedTree, new Filter<String>() {
			public boolean accept(String s) {
				return s.startsWith("Y");
			}
		});
		Tree<String> unAnnotatedTree = (new Trees.FunctionNodeStripper())
				.transformTree(debinarizedTree);
		return unAnnotatedTree;
	}

	public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {
		// Remove intermediate nodes (labels beginning with "@"
		// Remove all material on node labels which follow their base symbol
		// (cuts at the leftmost -, ^, or : character)
		// Examples: a node with label @NP->DT_JJ will be spliced out, and a
		// node with label NP^S will be reduced to NP
		Tree<String> debinarizedTree = debinarizeTree(annotatedTree);
		Tree<String> unAnnotatedTree = (new Trees.FunctionNodeStripper())
				.transformTree(debinarizedTree);
		return unAnnotatedTree;
	}

	private static Tree<String> debinarizeTree(Tree<String> annotatedTree) {
		return Trees.spliceNodes(annotatedTree, new Filter<String>() {
			public boolean accept(String s) {
				return s.startsWith("@") && s.length() > 1;
			}
		});
	}

	public static void main(String args[]) {
		// test the binarization
		Trees.PennTreeReader reader = new Trees.PennTreeReader(
				new StringReader(
						"((S (NP (DT the) (JJ quick) (JJ (AA (BB (CC brown)))) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))) (. .)))"));
		Tree<String> tree = reader.next();
		System.out.println("tree");
		System.out.println(Trees.PennTreeRenderer.render(tree));
		List<TreeBinarizer> values = Arrays.asList(new LeftBinarizer(), new RightBinarizer(),
				new HeadBinarizer(), new ParentBinarizer());
		for (TreeBinarizer binarization : values) {
			System.out.println("binarization type " + binarization.getClass().getSimpleName());
			// print the binarization
			try {
				Tree<String> binarizedTree = binarization.binarizeTree(tree);
				System.out.println(Trees.PennTreeRenderer.render(binarizedTree));
				System.out.println("unbinarized");
				Tree<String> unBinarizedTree = unAnnotateTree(binarizedTree);
				System.out.println(Trees.PennTreeRenderer.render(unBinarizedTree));
				System.out.println("------------");
			} catch (Error e) {
				System.out.println("binarization not implemented");
			}
		}
	}

	public static Tree<String> removeSuperfluousNodes(Tree<String> tree) {
		if (tree.isPreTerminal()) return tree;
		if (tree.isLeaf()) return tree;
		List<Tree<String>> gChildren = tree.getChildren();
		if (gChildren.size() != 1) {
			// nothing to do, just recurse
			ArrayList<Tree<String>> children = new ArrayList<Tree<String>>();
			for (int i = 0; i < gChildren.size(); i++) {
				Tree<String> cChild = removeSuperfluousNodes(tree.getChildren().get(i));
				children.add(cChild);
			}
			tree.setChildren(children);
			return tree;
		}
		Tree<String> result = null;
		String parent = tree.getLabel();
		HashSet<String> nodesInChain = new HashSet<String>();
		tree = tree.getChildren().get(0);
		while (!tree.isPreTerminal() && tree.getChildren().size() == 1) {
			if (!nodesInChain.contains(tree.getLabel())) {
				nodesInChain.add(tree.getLabel());
			}
			tree = tree.getChildren().get(0);
		}
		Tree<String> child = removeSuperfluousNodes(tree);
		String cLabel = child.getLabel();
		ArrayList<Tree<String>> childs = new ArrayList<Tree<String>>();
		childs.add(child);
		if (cLabel.equals(parent)) {
			result = child;
		} else {
			result = new Tree<String>(parent, childs);
		}
		for (String node : nodesInChain) {
			if (node.equals(parent) || node.equals(cLabel)) continue;
			Tree<String> intermediate = new Tree<String>(node, result.getChildren());
			childs = new ArrayList<Tree<String>>();
			childs.add(intermediate);
			result.setChildren(childs);
		}
		return result;
	}

	public static void displayUnaryChains(Tree<String> tree, String parent) {
		if (tree.getChildren().size() == 1) {
			if (!parent.equals("") && !tree.isPreTerminal())
				System.out.println("Unary chain: " + parent + " -> " + tree.getLabel() + " -> "
						+ tree.getChildren().get(0).getLabel());
			if (!tree.isPreTerminal())
				displayUnaryChains(tree.getChildren().get(0), tree.getLabel());
		} else {
			for (Tree<String> child : tree.getChildren()) {
				if (!child.isPreTerminal()) displayUnaryChains(child, "");
			}
		}

	}

	public static void removeUnaryChains(Tree<String> tree) {
		if (tree.isPreTerminal()) return;
		if (tree.getChildren().size() == 1
				&& tree.getChildren().get(0).getChildren().size() == 1) {
			// unary chain
			if (tree.getChildren().get(0).isPreTerminal())
				return; // if we are just above a preterminal, dont do anything
			else {// otherwise remove the intermediate node
				ArrayList<Tree<String>> newChildren = new ArrayList<Tree<String>>();
				newChildren.add(tree.getChildren().get(0).getChildren().get(0));
				tree.setChildren(newChildren);
			}
		}
		for (Tree<String> child : tree.getChildren()) {
			removeUnaryChains(child);
		}
	}

}
