package edu.berkeley.nlp.dep;

import java.util.Collection;

import edu.berkeley.nlp.syntax.Tree;

public interface TrainedDependencyScorer extends DependencyScorer {

	public void train(Collection<Tree<String>> trees);

}
