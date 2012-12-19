package edu.berkeley.nlp.dep;

import java.util.List;

public interface DependencyScorer {
	public void setInput(List<String> input);
	public double getDependencyScore(int head, int arg);
}
