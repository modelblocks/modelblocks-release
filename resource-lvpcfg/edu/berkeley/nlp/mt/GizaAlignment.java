package edu.berkeley.nlp.mt;

import java.util.List;

/**
 * GizaAlignments store additional information extracted from GIZA++ data files,
 * including the text, alignment score, sentence number, and filename.
 * 
 * @author John DeNero
 * 
 */
public class GizaAlignment extends Alignment {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double gizaScore;
	int gizaSentenceNumber;
	String gizaFilename;

	/**
	 * @return Returns the gizaFilename.
	 */
	public String getGizaFilename() {
		return gizaFilename;
	}

	/**
	 * @return Returns the gizaScore.
	 */
	public double getGizaScore() {
		return gizaScore;
	}

	/**
	 * @return Returns the gizaSentenceNumber.
	 */
	public double getGizaSentenceNumber() {
		return gizaSentenceNumber;
	}

	public GizaAlignment(List<String> targetSentence, List<String> sourceSentence,
			double gizaScore, int gizaSentenceNumber, String gizaFilename) {
		super(targetSentence, sourceSentence);
		this.gizaScore = gizaScore;
		this.gizaSentenceNumber = gizaSentenceNumber;
		this.gizaFilename = gizaFilename;
	}

	public GizaAlignment(List<String> targetSentence, List<String> sourceSentence) {
		super(targetSentence, sourceSentence);
	}

	/**
	 * Testing utility that outputs rendered alignment diagrams from a particular
	 * GIZA++ output file.
	 * 
	 * @param args
	 */
	public static void main(String args[]) throws Exception {
		// TODO: Check for argument and use other files.
		String testfile = "/Users/john/Documents/05Fall/cs294-5/corpora/alignments/sample-alignments";
		List<GizaAlignment> alignments = GizaAlignmentReader.readAlignments(testfile);
		for (GizaAlignment alignment : alignments) {
			System.out.println(alignment);
		}
	}

}