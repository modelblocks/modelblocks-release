package edu.berkeley.nlp.mt;

import java.io.Serializable;
import java.io.StringReader;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import fig.basic.ListUtils;
import fig.basic.StrUtils;

/**
 * A holder for a pair of sentences, each a list of strings. Sentences in the
 * test sets have integer IDs, as well, which are used to retreive the gold
 * standard alignments for those sentences.
 */
public class SentencePair implements Serializable {
	static final long serialVersionUID = 42;

	public int ID;
	String sourceFile;
	int lineNumber;
	List<String> englishWords, englishTags;
	List<String> foreignWords, foreignTags;
	Tree<String> englishTree;
	Tree<String> foreignTree;
	Alignment alignment;

	public SentencePair reverse() {
		SentencePair pair = new SentencePair(ID, sourceFile, lineNumber, foreignWords,
				englishWords);
		pair.foreignTags = englishTags;
		pair.englishTags = foreignTags;
		pair.foreignTree = englishTree;
		pair.englishTree = foreignTree;
		return pair;
	}

	public SentencePair(int sentenceID, String sourceFile, int lineNumber,
			List<String> englishWords, List<String> frenchWords) {
		this.ID = sentenceID;
		this.sourceFile = sourceFile;
		this.lineNumber = lineNumber;
		this.englishWords = englishWords;
		this.foreignWords = frenchWords;
	}

	public int getSentenceID() {
		return ID;
	}

	public String getSourceFile() {
		return sourceFile;
	}

	public List<String> getEnglishWords() {
		return englishWords;
	}

	public List<String> getForeignWords() {
		return foreignWords;
	}

	public int I() {
		return englishWords.size();
	}

	public int J() {
		return foreignWords.size();
	}

	public String en(int i) {
		return englishWords.get(i);
	}

	public String fr(int j) {
		return foreignWords.get(j);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int englishPosition = 0; englishPosition < englishWords.size(); englishPosition++) {
			String englishWord = englishWords.get(englishPosition);
			sb.append(englishPosition);
			sb.append(":");
			sb.append(englishWord);
			sb.append(" ");
		}
		sb.append("\n");
		for (int frenchPosition = 0; frenchPosition < foreignWords.size(); frenchPosition++) {
			String frenchWord = foreignWords.get(frenchPosition);
			sb.append(frenchPosition);
			sb.append(":");
			sb.append(frenchWord);
			sb.append(" ");
		}
		sb.append("\n");
		return sb.toString();
	}

	// Return the set of words used in these sentences.
	public static Set<String> getWordSet(List<SentencePair> sentencePairs, boolean isForeign) {
		Set<String> set = new HashSet<String>();
		for (SentencePair sp : sentencePairs) {
			List<String> words = isForeign ? sp.getForeignWords() : sp.getEnglishWords();
			for (String w : words)
				set.add(w);
		}
		return set;
	}

	public SentencePair chop(int i1, int i2, int j1, int j2) {
		return new SentencePair(ID, sourceFile, lineNumber, englishWords.subList(i1, i2),
				foreignWords.subList(j1, j2));
	}

	public Tree<String> getEnglishTree() {
		return englishTree;
	}

	public void setEnglishTree(Tree<String> englishTree) {
		this.englishTree = englishTree;
	}

	public Tree<String> getForeignTree() {
		return foreignTree;
	}

	public void setForeignTree(Tree<String> frenchTree) {
		this.foreignTree = frenchTree;
	}

	public Alignment getAlignment() {
		return alignment;
	}

	public void setAlignment(Alignment referenceAlignment) {
		this.alignment = referenceAlignment;
	}

	public List<String> getEnglishTags() {
		return englishTags;
	}

	public void setEnglishTags(List<String> englishTags) {
		this.englishTags = englishTags;
	}

	public List<String> getForeignTags() {
		return foreignTags;
	}

	public void setForeignTags(List<String> foreignTags) {
		this.foreignTags = foreignTags;
	}

	public String dump() {
		StringBuffer sbuf = new StringBuffer();
		sbuf.append("ID:\t" + ID + "\tSource file:\t" + sourceFile + "\n");
		sbuf.append("En:\t" + StrUtils.join(englishWords, " ") + "\n");
		sbuf.append("Fr:\t" + StrUtils.join(foreignWords, " ") + "\n");
		sbuf.append("EnTags:\t");
		sbuf.append(englishTags);
		sbuf.append("\n");
		sbuf.append("FrTags:\t");
		sbuf.append(foreignTags);
		sbuf.append("\n");
		sbuf.append("EnTree:\t");
		sbuf.append(englishTree);
		sbuf.append("\n");
		sbuf.append("FrTree:\t");
		sbuf.append(foreignTree);
		sbuf.append("\n");
		sbuf.append("Alignment:\n");
		sbuf.append(alignment);
		return sbuf.toString();
	}

	public static SentencePair getSampleSentencePair() {
		String p = "(S (NP (DT the) (NNS jobs)) (VP (VBP are) (ADJP (NN career) (VBN oriented))) (. .))";
		Trees.PennTreeReader treeReader = new Trees.PennTreeReader(new StringReader(p));
		Tree<String> tree = treeReader.next();
		List<String> en = tree.getYield();
		List<String> fr = ListUtils.newList("les", "emplois", "sont", "axes", "sur", "la",
				"carriere", ".");
		SentencePair sp = new SentencePair(0, "", 0, en, fr);
		sp.setEnglishTree(tree);
		Alignment a = new Alignment(en, fr);
		a.addAlignment(0, 0);
		a.addAlignment(1, 1);
		a.addAlignment(2, 2);
		a.addAlignment(3, 6);
		a.addAlignment(4, 3);
		a.addAlignment(5, 7);
		//		a.addAlignment(0, 5);
		sp.setAlignment(a);
		return sp;
	}

	public void setLineNumber(int sent) {
		lineNumber = sent;
	}
}
