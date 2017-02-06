package edu.berkeley.nlp.mt;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author denero
 *
 * @param <T>
 */
public class SubSequenceMatcher<T> {

	Set<List<T>> subsequences = new HashSet<List<T>>();
	
	public void addSequence(List<T> sequence) {
		for (int i = 0; i < sequence.size(); i++) {
			for (int j = i + 1; j <= sequence.size(); j++) {
				subsequences.add(sequence.subList(i, j));
			}
		}
	}
	
	public boolean containsSubSequence(List<T> subsequence) {
		return subsequences.contains(subsequence);
	}

}
