package edu.berkeley.nlp.dep;

import java.util.List;

public class PosteriorDependencyScorer implements DependencyScorer {
	ExhaustiveDependencyParser depParser ;
	double[][] depPosteriors ;

	public PosteriorDependencyScorer(DependencyScorer depScorer) {
		depParser = new ExhaustiveDependencyParser(depScorer);
	}

	public double getDependencyScore(int head, int arg) {
		return depPosteriors[head][arg] ;
	}
	public void setInput(List<String> input) {
		depParser.setInput(input);
		depPosteriors = depParser.getDependencyPosteriors();
	}

}
