package edu.berkeley.nlp.dep;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.ling.HeadFinder;
import edu.berkeley.nlp.syntax.Tree;

/**
 * Basic dependency structure as defined in Shen, Xu and Weishedel 2008 (ACL MT paper).
 * 
 * @author dburkett
 * 
 * So far, this just creates a vector for an entire tree and does not support partial structures.
 */

public class DependencyStructure {
	private List<String> words;
	private List<Integer> heads;

	public DependencyStructure(Tree<String> tree, HeadFinder hf) {
		words = tree.getYield();
		heads = new ArrayList<Integer>(words.size());
		for (int i=0; i<words.size(); i++) {
			heads.add(-1);
		}
		Tree<Integer> depTree = new DependencyTreeExtractor(hf).getDependencyTreeIndices(tree);
		recordHeads(depTree, -1);
	}

	private void recordHeads(Tree<Integer> node, int parentIndex) {
		int index = node.getLabel();
		heads.set(index, parentIndex);
		for (Tree<Integer> child : node.getChildren()) {
			recordHeads(child, index);
		}
	}
	
	@Override
	public String toString() {
		return listToString(heads);
	}
	
	private static <T> String listToString(List<T> items) {
		StringBuffer sb = new StringBuffer();
		for (T item : items) {
			sb.append(item + " ");
		}
		return sb.toString().trim();
	}
}
