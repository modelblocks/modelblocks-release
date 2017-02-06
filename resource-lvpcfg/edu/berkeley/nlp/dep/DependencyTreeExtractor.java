package edu.berkeley.nlp.dep;

import java.io.StringReader;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.ling.CollinsHeadFinder;
import edu.berkeley.nlp.ling.HeadFinder;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees.PennTreeReader;

public class DependencyTreeExtractor {
	
	private final HeadFinder headFinder;
	
	public DependencyTreeExtractor(HeadFinder headFinder) {
		this.headFinder = headFinder;
	}
	
	private Tree<String> getHeadTerminal(Tree<String> tree) {
		Tree<String> child = headFinder.determineHead(tree);
		if (child.isLeaf()) {
			return child;
		}
		return getHeadTerminal(child);
	}
	
	private Tree<Tree<String>> getDependencyTreeRec(Tree<String> tree) {
		
		if (tree.isPreTerminal())	 {
			return new Tree<Tree<String>>(tree.getChildren().get(0));
		}
		
		Tree<String> headLeaf = getHeadTerminal(tree);
		assert headLeaf.isLeaf();
		
		List<Tree<Tree<String>>> children = new ArrayList<Tree<Tree<String>>>();
		for (Tree<String> child: tree.getChildren()) {
			children.add( getDependencyTreeRec(child) );
		}
		
		List<Tree<Tree<String>>> flatChildren = new ArrayList<Tree<Tree<String>>>(); 
		for (int i=0; i < children.size(); ++i) {
			Tree<Tree<String>> headChildTree = children.get(i);
			Tree<String> headChild = headChildTree.getLabel();
			if (headChild == headLeaf) {
				flatChildren.addAll(headChildTree.getChildren());
			} else {
				flatChildren.add(headChildTree);
			}
		}
		
		return new Tree<Tree<String>>(headLeaf, flatChildren);
	}
	
	
	/**
	 * 
	 * @param tree
	 * @return
	 */
	public Tree<Integer> getDependencyTreeIndices(Tree<String> tree) {
		List<Tree<String>> leaves = tree.getTerminals();
		Map<Tree<String>, Integer> leafIndexMap = new IdentityHashMap<Tree<String>, Integer>();
		for (int i=0; i < leaves.size(); ++i) {
			Tree<String> leaf = leaves.get(i);
			assert leaf.isLeaf();
			leafIndexMap.put(leaf, i);
		}
		Tree<Tree<String>> depTree = getDependencyTreeRec(tree);
//		System.out.println(depTree);
//		System.out.println(leafIndexMap);
		return convert(depTree,leafIndexMap);
	}
	
	
	
	private static <S,T> Tree<T> convert(Tree<S> tree, Map<S,T> labelMap) {
		S label = tree.getLabel();
		T transLabel = labelMap.get(label);
		List<Tree<T>> children = new ArrayList<Tree<T>>();
		for (Tree<S> child: tree.getChildren()) {
			children.add( convert(child, labelMap) );			
		}
		return new Tree<T>(transLabel, children);
	}
	
	public Tree<String> getDependencyTree(Tree<String> tree) {
		List<Tree<String>> leaves = tree.getTerminals();
		Map<Integer, String> leafMap = new HashMap<Integer, String>();
		for (int i=0; i < leaves.size(); ++i) {
			leafMap.put(i, leaves.get(i).getLabel());
		}
//		System.out.println(leafMap);
		return convert(getDependencyTreeIndices(tree), leafMap);
	}
	
	
	private static Tree<String> getTree(String treeStr) {
		return (new PennTreeReader(new StringReader(treeStr))).next();
	}
	
	public static void main(String[] args) {
		Tree<String> tree = getTree("((S (NP (DT the) (JJ quick) (JJ (AA (BB (CC brown)))) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))) (. .)))");
		DependencyTreeExtractor depExtractor = new DependencyTreeExtractor(new CollinsHeadFinder());
		System.out.println(depExtractor.getDependencyTree(tree));
	}

}
