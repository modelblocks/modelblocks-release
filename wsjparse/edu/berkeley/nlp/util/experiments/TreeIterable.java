package edu.berkeley.nlp.util.experiments;

import java.io.File;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import edu.berkeley.nlp.syntax.Trees.TreeTransformer;
import edu.berkeley.nlp.util.Filter;
import edu.berkeley.nlp.util.IOUtil;
import edu.berkeley.nlp.util.Iterators;
import edu.berkeley.nlp.util.Logger;
import edu.berkeley.nlp.util.Method;
import fig.basic.IOUtils;
import fig.basic.Option;
import fig.exec.Execution;

/**
 * 
 * 
 * @author aria42
 *
 */
public class TreeIterable implements Iterable<Tree<String>>, Runnable {
	
	@Option
	public String dataRoot ;
	@Option
	public String extension = "mrg";
	@Option
	public boolean doStandardNormalization = true;
	@Option
	public int minLength = 0;
	@Option
	public int maxLength = Integer.MAX_VALUE;
	@Option	
	public int maxNumTrees = Integer.MAX_VALUE;
	@Option
	public boolean lowercase = false;
	@Option
	public String prefix = "";
	@Option
	public boolean stripPunctuation = false;
	
	private TreeTransformer<String> treeTransformer = null;
		
	private void initializeTreeTransformer( ) {
		Trees.CompoundTreeTransformer<String> treeTransformer = new Trees.CompoundTreeTransformer();
		if (doStandardNormalization) {
			treeTransformer.addTransformer(new Trees.StandardTreeNormalizer());
		}
		if (stripPunctuation) {
			treeTransformer.addTransformer(new Trees.PunctuationStripper());
		}
		this.treeTransformer = treeTransformer;
	}
	
	public class MyIterator implements Iterator<Tree<String>> {
		
		private Iterator<File> fileIt;
		private Iterator<Tree<String>> treeIt = Iterators.emptyIterator();
		public MyIterator(Iterator<File> fileIt) {
			this.fileIt = fileIt;
		}
		
		private Iterator<Tree<String>> getTreeIterator( ) { 
			String nextPath = fileIt.next().getAbsolutePath();
			Iterator<Tree<String>> treeIt = new Trees.PennTreeReader(IOUtils.openInHard(nextPath));
			Method<Tree<String>, Tree<String>> transformMethod = new Method<Tree<String>, Tree<String>>() {
				public Tree<String> call(Tree<String> obj) {
					return treeTransformer.transformTree(obj);
				}				
			};
			Filter<Tree<String>> treeFilter = new Filter<Tree<String>>() {
				public boolean accept(Tree<String> t) {
					int length = t.getYield().size();
					return (length >= minLength && length <= maxLength);
				}				
			};
			treeIt = new Iterators.TransformingIterator<Tree<String>, Tree<String>>(treeIt,transformMethod) ;			
			treeIt = new Iterators.FilteredIterator<Tree<String>>(treeFilter,treeIt);
			return treeIt;
		}
		
		private boolean queueNext() {
			if (treeIt.hasNext()) {
				return true;
			}
			if (!fileIt.hasNext()) {
				return false;
			}			
			treeIt = getTreeIterator();
			return queueNext();
		}		
		
		public boolean hasNext() {
			return queueNext();
		}

		public Tree<String> next() {
			queueNext();			
			return treeTransformer.transformTree(treeIt.next());
		}

		public void remove() {
			throw new UnsupportedOperationException();
		}		
	}
		
	public Iterator<Tree<String>> iterator() {
		List<File> files = IOUtils.getFilesUnder(dataRoot, IOUtil.getFileFilter(prefix, extension));
		initializeTreeTransformer();
		Iterator<Tree<String>> treeIt = new MyIterator(files.iterator());
		return Iterators.maxLengthIterator(treeIt, maxNumTrees);
	}
	
	public void run() {
		int count = 0;
		for (Tree<String> t: this) {
			System.out.println(t.toString());
			++count;
		}
		Logger.i().logs("Number of Trees: %d", count);
	}
	
	public static void main(String[] args) {
		Execution.run(args, new TreeIterable());
	}
}
