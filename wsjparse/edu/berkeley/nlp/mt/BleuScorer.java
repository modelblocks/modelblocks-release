package edu.berkeley.nlp.mt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.Iterators;
import fig.basic.IOUtils;
import fig.basic.StrUtils;

/**
 * A bleu score evaluator, as specified in:
 * <a href="http://www1.cs.columbia.edu/nlp/sgd/bleu.pdf">http://www1.cs.columbia.edu/nlp/sgd/bleu.pdf</a>
 *
 * @author Alexandre Bouchard
 */
public class BleuScorer {
	/**
	 * The maximum length of the n-gram considered
	 */
	private int N;

	/**
	 * The weights used to combine the individual n-grams
	 */
	private List<Double> weights;

	/**
	 * The setting usually preferred for reporting are 
	 * N=4 with the weights uniform.
	 */
	public BleuScorer() {
		this(4);
	}

	public BleuScorer(int N) {
		this.N = N;
		weights = new ArrayList<Double>();
		for (int i = 0; i < N; i++) {
			weights.add((double) 1 / (double) N);
		}
	}

	/**
	 * If weights == null, this is equivalent to calling the anonymous constructor.
	 * @param weights The weigths used to combine the individual n-grams
	 */
	public BleuScorer(List<Double> weights) {
		this();
		if (weights != null) {
			this.weights = weights;
			this.N = weights.size();
		}
	}

	private static List<String> normalizeText(List<String> words) {
		String text = StrUtils.join(words);

		// We assume most of the tokenization has already been done properly

		// Language-independent part:
		//text  =~ s/<skipped>//g; # strip "skipped" tags
		//text  =~ s/-\n//g; # strip end-of-line hyphenation and join lines
		//text  =~ s/\n/ /g; # join lines
		text = text.replaceAll("(\\d)\\s+(?=\\d)", "$1"); // join digits
		//text  =~ s/&quot;/"/g;  # convert SGML tag for quote to "
		//text  =~ s/&amp;/&/g;   # convert SGML tag for ampersand to &
		//text  =~ s/&lt;/</g;    # convert SGML tag for less-than to >
		//text  =~ s/&gt;/>/g;    # convert SGML tag for greater-than to <
		// Language-dependent part (assuming Western languages):
		//text  = " $norm_text ";
		//text  =~ tr/[A-Z]/[a-z]/ unless $preserve_case;
		text = text.replaceAll("([\\{-\\~\\[-\\` -\\&\\(-\\+\\:-\\@\\/])", " $1 "); // tokenize punctuation
		text = text.replaceAll("([^0-9])([\\.,])", "$1 $2 "); // tokenize period and comma unless preceded by a digit
		text = text.replaceAll("([\\.,])([^0-9])", " $1 $2"); // tokenize period and comma unless followed by a digit
		text = text.replaceAll("([0-9])(-)", "$1 $2 "); // tokenize dash when preceded by a digit
		//text  =~ s/\s+/ /g; # one space only between words
		//text  =~ s/^\s+//;  # no leading space
		//text  =~ s/\s+$//;  # no trailing space
		words = Arrays.asList(StrUtils.split(text.trim(), "\\s+"));
		//fig.basic.LogInfo.stderr.println(StrUtils.join(words));
		return words;
	}

	/**
	 * Evaluate a bleu score. 
	 * 
	 * The nesting of the lists has the following meaning:
	 * Define a Sentence as a List of String's, <br/>
	 * The parameter candidates should be a List of Sentence's <br/>
	 * Define a Reference as a List of Sentence's <br/>
	 * The parameter referenceSet should be a List of Reference's
	 * 
	 * TODO: integrate the string transformations that the nist perl script uses:
	 * # language-independent part:
	 * $norm_text =~ s/<skipped>//g; # strip "skipped" tags
	 * $norm_text =~ s/-\n//g; # strip end-of-line hyphenation and join lines
	 * $norm_text =~ s/\n/ /g; # join lines
	 * $norm_text =~ s/(\d)\s+(?=\d)/$1/g; #join digits
	 * $norm_text =~ s/&quot;/"/g;  # convert SGML tag for quote to "
	 * $norm_text =~ s/&amp;/&/g;   # convert SGML tag for ampersand to &
	 * $norm_text =~ s/&lt;/</g;    # convert SGML tag for less-than to >
	 * $norm_text =~ s/&gt;/>/g;    # convert SGML tag for greater-than to <
	 * 
	 * # language-dependent part (assuming Western languages):
	 * $norm_text = " $norm_text ";
	 * $norm_text =~ tr/[A-Z]/[a-z]/ unless $preserve_case;
	 * $norm_text =~ s/([\{-\~\[-\` -\&\(-\+\:-\@\/])/ $1 /g;   # tokenize punctuation
	 * $norm_text =~ s/([^0-9])([\.,])/$1 $2 /g; # tokenize period and comma unless preceded by a digit
	 * $norm_text =~ s/([\.,])([^0-9])/ $1 $2/g; # tokenize period and comma unless followed by a digit
	 * $norm_text =~ s/([0-9])(-)/$1 $2 /g; # tokenize dash when preceded by a digit
	 * $norm_text =~ s/\s+/ /g; # one space only between words
	 * $norm_text =~ s/^\s+//;  # no leading space
	 * $norm_text =~ s/\s+$//;  # no trailing space
	 * 
	 * @param candidates A list of sentence translated by the mt system.
	 * @param testSentences A list of set of "referenceSet"s.
	 * @return
	 */
	public BleuScore evaluateBleu(List<List<String>> candidates,
			List<TestSentence> testSentences, boolean normalize) {
		if (normalize) {
			List<List<String>> newCandidates = new ArrayList<List<String>>();
			for (List<String> candidate : candidates)
				newCandidates.add(normalizeText(candidate));
			candidates = newCandidates;

			List<TestSentence> newTestSentences = new ArrayList<TestSentence>();
			for (TestSentence testSentence : testSentences) {
				List<List<String>> newReferenceSet = new ArrayList<List<String>>();
				for (List<String> reference : testSentence.getReferences())
					newReferenceSet.add(normalizeText(reference));
				newTestSentences.add(new TestSentence(testSentence.getForeignSentence(),
						newReferenceSet));
			}
			testSentences = newTestSentences;
		}

		List<Double> individualNGramScorings = new ArrayList<Double>();
		for (int i = 0; i < N; i++) {
			individualNGramScorings.add(computeIndividualNGramScoring(i + 1, candidates,
					testSentences));
		}
		return new BleuScore(individualNGramScorings, weights, computeR(candidates,
				testSentences), computeC(candidates));
	}

	/**
	 * c is the total length of the candidate translation corpus.
	 * 
	 * @param candidates
	 * @return
	 */
	protected double computeC(List<List<String>> candidates) {
		double sum = 0.0;
		for (List<String> currentCandidate : candidates) {
			sum += currentCandidate.size();
		}
		return sum;
	}

	/**
	 * 
	 * The test corpus effective reference length, r, is computed by summing the best match lengths 
	 * for each candidate sentence in the corpus.
	 * 
	 * @param candidates
	 * @param testSentences
	 * @return
	 */
	protected double computeR(List<List<String>> candidates,
			List<TestSentence> testSentences) {
		double sum = 0.0;
		for (int i = 0; i < candidates.size(); i++) {
			double min = Double.POSITIVE_INFINITY;
			double argmin = 0.0;
			// find the best match
			for (List<String> reference : testSentences.get(i).getReferences()) {
				double currentValue = Math.abs(reference.size() - candidates.get(i).size());
				if (currentValue < min) {
					min = currentValue;
					argmin = reference.size();
				}
			}
			sum += argmin;
		}
		return sum;
	}

	/**
	 * 
	 * Compute the modified unigram precisions. 
	 * 
	 * To compute this,
	 * one first counts the maximum number of times
	 * a word occurs in any single reference translation.
	 * Next, one clips the total count of each candidate
	 * word by its maximum reference count,
	 * adds these clipped counts up, and divides by the
	 * total (unclipped) number of candidate words.
	 * Modified n-gram precision is computed similarly
	 * for any n: all candidate n-gram counts
	 * and their corresponding maximum reference
	 * counts are collected. The candidate counts are
	 * clipped by their corresponding reference maximum
	 * value, summed, and divided by the total
	 * number of candidate n-grams.
	 * How do we compute modified n-gram precision
	 * on a multi-sentence test set? Although one typically
	 * evaluates MT systems on a corpus of entire
	 * documents, our basic unit of evaluation is the
	 * sentence. A source sentence may translate to
	 * many target sentences, in which case we abuse
	 * terminology and refer to the corresponding target
	 * sentences as a sentence. We first compute
	 * the n-gram matches sentence by sentence.
	 * Next, we add the clipped n-gram counts for all
	 * the candidate sentences and divide by the number
	 * of candidate n-grams in the test corpus to 
	 * compute a modified precision score, pn, for the
	 * entire test corpus.
	 * In other words, we use a word-weighted average
	 * of the sentence-level modified precisions rather
	 * than a sentence-weighted average. As an example,
	 * we compute word matches at the sentence
	 * level, but the modified unigram precision is the
	 * fraction of words matched in the entire test corpus.
	 * 
	 * @param n n in n-gram
	 * @param candidates 
	 * @param testSentences
	 * @return
	 */
	protected double computeIndividualNGramScoring(int n, List<List<String>> candidates,
			List<TestSentence> testSentences) {
		double denominator = 0.0;
		double numerator = 0.0;
		// loop over all (candidate, referenceSet) pairs
		for (int i = 0; i < candidates.size(); i++) {
			List<String> currentCandidate = candidates.get(i);
			// extract the counts of all the k-grams, where k=n....
			// ...in the candidate...
			Counter<List<String>> candidateNGramCounts = extractNGramCounts(n, currentCandidate);
			// ...and in each of the references
			List<Counter<List<String>>> referenceSetNGramCounts = new ArrayList<Counter<List<String>>>();
			for (List<String> reference : testSentences.get(i).getReferences()) {
				referenceSetNGramCounts.add(extractNGramCounts(n, reference));
			}
			// compute the modified n-gram precisions 
			for (List<String> currentNGram : candidateNGramCounts.keySet()) {
				// the count in the candidate sentence of the current n-gram is added to the denominator
				double currentCount = candidateNGramCounts.getCount(currentNGram);
				denominator += currentCount;
				// find, over all the references, the maximum number of occurrence of the current ngram 
				double max = 0.0;
				for (Counter<List<String>> currentReferenceNGramCounts : referenceSetNGramCounts) {
					double tempCount = currentReferenceNGramCounts.getCount(currentNGram);
					if (tempCount > max) {
						max = tempCount;
					}
				}
				// the minimum of {max, currentCount} is added to the numerator
				if (max < currentCount) {
					numerator += max;
				} else {
					numerator += currentCount;
				}
			}
		}
		// if the sums were empty, return 0.0 (to mirror NIST standard)
		if (denominator == 0.0) {
			return 0.0;
		} else {
			return numerator / denominator;
		}
	}
	
	public static void main(String[] argv)
	{
		if (argv.length < 2 || argv.length > 3)
		{
			System.err.println("Args: [translation file] [reference file] <max length of reference sentences>.");
			System.exit(1);
		}
//		 CorrespondingIterable<String> pairIterable = new CorrespondingIterable<String>(Iterators.able(IOUtils.lineIterator(argv[0])), Iterators.able(IOUtils.lineIterator(argv[1])));
		 BleuScorer scorer = new BleuScorer();
		 List<TestSentence> references = new ArrayList<TestSentence>();
		 List<List<String>> candidates = new ArrayList<List<String>>();
		 int maxLength = argv.length == 3 ? Integer.parseInt(argv[2]) : Integer.MAX_VALUE;
		Set<Integer> filtered = new HashSet<Integer>();
		 
		 try
		{
			 int x = 0;
			for (String line : Iterators.able(IOUtils.lineIterator(argv[1])))
			{
				
				final List<String> asList = Arrays.asList(line.split(" "));
				final TestSentence o = new TestSentence(Collections.singletonList("dummy"), Collections.singletonList(asList));
				if (asList.size() > maxLength)
				{
					filtered.add(x);
				}
				else
				{
					references.add(o);
				}
				x++;
			}
			int y = 0;
			for (String line : Iterators.able(IOUtils.lineIterator(argv[0])))
			{
				if (!filtered.contains(y)) 
					candidates.add(Arrays.asList(line.split(" ")));
				y++;
			}
		}
		catch (IOException e)
		{
			// TODO Auto-generated catch block
			throw new RuntimeException(e);

		}

		if (candidates.size() != references.size())
			throw new RuntimeException("Reference length = " + references.size() + " and candidate length = " + candidates.size());
			 
		BleuScore score = scorer.evaluateBleu(candidates, references, true);
		System.out.println("Bleu score is " + score);
		System.exit(1);
			 
	}

	/**
	 * Extract all the ngrams and their counts in a given sentence.
	 * 
	 * @param n n in n-gram
	 * @param sentences
	 * @return
	 */
	protected Counter<List<String>> extractNGramCounts(int n, List<String> sentences) {
		Counter<List<String>> nGrams = new Counter<List<String>>();
		for (int i = 0; i <= sentences.size() - n; i++) {
			nGrams.incrementCount(sentences.subList(i, i + n), 1.0);
		}
		return nGrams;
	}

}
