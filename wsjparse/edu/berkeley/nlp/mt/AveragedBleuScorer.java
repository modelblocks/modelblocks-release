/**
 * 
 */
package edu.berkeley.nlp.mt;

import java.util.List;


/**
 * @author Alexandre Bouchard
 *
 */
public class AveragedBleuScorer
{	
	/**
	 * 
	 */
	private BleuScorer baseScorer;
	
	/**
	 * 
	 * @param maxN
	 */
	public AveragedBleuScorer(int maxN)
	{
		baseScorer = new BleuScorer(maxN);
	}
	
	/**
	 * 
	 *
	 */
	public AveragedBleuScorer()
	{
		this(4);
	}

	/**
	 * 
	 * @param candidates
	 * @param references
	 * @param normalize
	 * @return
	 */
	public AveragedBleuScore evaluateBleu(List<List<String>> candidates,
			List<TestSentence> testSentences, boolean normalize)
	{
		return new AveragedBleuScore(baseScorer.evaluateBleu(candidates, testSentences, normalize));
	}
	
//	/**
//	 * 
//	 * @param candidate
//	 * @param reference
//	 * @return
//	 */
//	public <T extends SentencePairExample> Score scoreSentencePairExample(T candidate, T reference)
//	{
//		return new AveragedBleuScore(baseScorer.scoreSentencePairExample(candidate, reference));
//	}
//
//	/**
//	 * 
//	 * @param candidates
//	 * @param referenceSets
//	 * @return
//	 */
//	public <T extends SentencePairExample> Score scoreSentencePairExamples(List<T> candidates, List<List<T>> referenceSets)
//	{
//		return new AveragedBleuScore(baseScorer.scoreSentencePairExamples(candidates, referenceSets));
//	}
}
