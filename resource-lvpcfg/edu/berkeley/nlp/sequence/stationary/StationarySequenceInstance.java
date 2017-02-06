package edu.berkeley.nlp.sequence.stationary;

public interface StationarySequenceInstance {
	/**
	 * What is the length of the instance?
	 * @return
	 */
	public int getSequenceLength();
	/**
	 * Put the score for giving position i label s
	 * You should only fill in [getSequenceLength(), numStates]
	 * portion of the array
	 * @param potentials
	 */
	public void fillNodePotentials(double[][] potentials);
	
}
