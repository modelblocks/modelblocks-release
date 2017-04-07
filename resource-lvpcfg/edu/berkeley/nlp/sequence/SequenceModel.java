package edu.berkeley.nlp.sequence;

public interface SequenceModel {


	/**
	 * How many label states are there?
	 * @return
	 */
	public int getNumLabels();
	/**
	 * What is the maximum seq. length
	 * @return
	 */
	public int getMaximumSequenceLength() ;

	/**
	 * For each state which states can follow it?
	 * @return
	 */
	public int[][] getAllowableForwardTransitions();
	/**
	 * For each state what can pre-cede it
	 * @return
	 */
	public int[][] getAllowableBackwardTransitions();
}
