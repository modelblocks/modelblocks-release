/**
 * 
 */
package edu.berkeley.nlp.PCFGLA;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.discPCFG.DefaultLinearizer;
import edu.berkeley.nlp.discPCFG.Linearizer;
import edu.berkeley.nlp.syntax.StateSet;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.util.ArrayUtil;
import edu.berkeley.nlp.util.Numberer;
import fig.basic.Pair;
import edu.berkeley.nlp.util.ScalingTools;

/**
 * @author adpauls
 * 
 */
public class PosteriorConstrainedTwoChartsParser extends
		ConstrainedTwoChartsParser {

	protected double[][][][] iPosteriorScorePreU, iPosteriorScorePostU;

	protected double[][][][] oPosteriorScorePreU, oPosteriorScorePostU;

	protected int[][][] iPosteriorScale;

	protected int[][][] oPosteriorScale;

	private Tree<StateSet> goldTree;

	private boolean[][][] outsideAlreadyAdded;

	private double[] scoresToAdd1;

	private double[] scoresToAdd2;

	private double[] unscaledScoresToAdd1;

	private double[] unscaledScoresToAdd2;

	private double[][][] posteriors;

	// private boolean[][][] insideAlreadyAdded;

	/**
	 * @param gr
	 * @param lex
	 * @param boost
	 */
	public PosteriorConstrainedTwoChartsParser(Grammar gr, Lexicon lex,
			double boost) {
		super(gr, lex, null);
		scoresToAdd1 = new double[(int)ArrayUtil.max(numSubStatesArray)];
		unscaledScoresToAdd1 = new double[scoresToAdd1.length];
		scoresToAdd2 = new double[(int)ArrayUtil.max(numSubStatesArray)];
		unscaledScoresToAdd2 = new double[scoresToAdd1.length];
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param state
	 * @return
	 */
	public double getSummedPosterior(short state, int start, int end,
			boolean isUnaryTop, boolean isUnaryBottom) {
		int totalStates = 0, previouslyPossible = 0, nowPossible = 0;

		double sentenceProb = iScorePostU[0][length][0][0];
		double sentenceScale = iScale[0][length][0];
		double totalPosterior = 0.0;
		// if (level<1) nowPossible=totalStates=previouslyPossible=length;
		// int startDiff = (level<0) ? 2 : 1;
		// int startDiff = 1;
		// for (int diff = startDiff; diff <= length; diff++) {
		// for (int start = i; start < (length - diff + 1); start++) {

		int lastState = getNumSubStatesArray().length;

		if (start - end > 1 && !grammarTags[state])
			return 0.0;

		if (allowedSubStates[start][end][state] == null)
			return 0.0;
		boolean nonePossible = true;
		int thisScale = iScale[start][end][state] + oScale[start][end][state];
		double scalingFactor = 1;
		if (thisScale != sentenceScale) {
			scalingFactor *= Math.pow(ScalingTools.SCALE, thisScale - sentenceScale);
		}
		if (Double.isInfinite(scalingFactor))
		{
			System.out.println("possible overflow");
		}

		for (int substate = 0; substate < getNumSubStatesArray()[state]; substate++) {
			totalStates++;
			if (!allowedSubStates[start][end][state][substate])
				continue;
			if (iScorePostU[start][end][state] == null)
				continue;
			previouslyPossible++;
			double iS = (isUnaryBottom && !isUnaryTop) ? iScorePreU[start][end][state][substate]
					: iScorePostU[start][end][state][substate];
			double oS = (!isUnaryBottom && isUnaryTop) ? oScorePreU[start][end][state][substate]
					: oScorePostU[start][end][state][substate];

			if (iS == 0 || oS == 0) {
//				assert false;
				allowedSubStates[start][end][state][substate] = false;
				continue;
			}

			final double d = iS * scalingFactor * oS / sentenceProb;
			if (iS > 0.0 && oS > 0.0 && scalingFactor != 0.0 && d == 0.0)
			{
				System.out.println("possible underflow");
			}
			assert d <1.01;
			
			totalPosterior += d;
		}

		return totalPosterior;
		// if (posterior > threshold) {
		// allowedSubStates[start][end][state][substate]=true;
		// nowPossible++;
		// nonePossible=false;
		// } else {
		// allowedSubStates[start][end][state][substate] = false;
		// }

		// if (nonePossible) allowedSubStates[start][end][state] = null;

	}

	/**
	 * @param state
	 * @return
	 */
	public double getLogSummedPosterior(short state, int start, int end,
			boolean isUnaryTop, boolean isUnaryBottom) {
		int totalStates = 0, previouslyPossible = 0, nowPossible = 0;

		double sentenceProb = iScorePostU[0][length][0][0];
		double sentenceScale = iScale[0][length][0];
		double totalPosterior = 0.0;
		// if (level<1) nowPossible=totalStates=previouslyPossible=length;
		// int startDiff = (level<0) ? 2 : 1;
		// int startDiff = 1;
		// for (int diff = startDiff; diff <= length; diff++) {
		// for (int start = i; start < (length - diff + 1); start++) {

		int lastState = getNumSubStatesArray().length;

		if (start - end > 1 && !grammarTags[state])
			return Double.NEGATIVE_INFINITY;

		if (allowedSubStates[start][end][state] == null)
			return Double.NEGATIVE_INFINITY;
		boolean nonePossible = true;
		int thisScale = iScale[start][end][state] + oScale[start][end][state];
		double scalingFactor = 1;
//		if (thisScale != sentenceScale) {
//			scalingFactor *= Math.pow(ScalingTools.SCALE, thisScale - sentenceScale);
//		}
		scalingFactor = (thisScale-sentenceScale) * Math.log(ScalingTools.SCALE);
//		if (Double.isInfinite(scalingFactor))
//		{
//			System.out.println("possible overflow");
//		}
		double[] tmp = new double[ getNumSubStatesArray()[state]];
		Arrays.fill(tmp, Double.NEGATIVE_INFINITY);
		for (int substate = 0; substate < getNumSubStatesArray()[state]; substate++) {
			totalStates++;
			if (!allowedSubStates[start][end][state][substate])
				continue;
			if (iScorePostU[start][end][state] == null)
				continue;
			previouslyPossible++;
			double iS = (isUnaryBottom && !isUnaryTop) ? iScorePreU[start][end][state][substate]
					: iScorePostU[start][end][state][substate];
			double oS = (!isUnaryBottom && isUnaryTop) ? oScorePreU[start][end][state][substate]
					: oScorePostU[start][end][state][substate];

			if (iS == 0 || oS == 0) {
//				assert false;
				allowedSubStates[start][end][state][substate] = false;
				continue;
			}

			final double d = Math.log(iS) + Math.log(oS) - Math.log(sentenceProb) + scalingFactor;
			assert d < 0.01;
//			final double d = iS * scalingFactor * oS / sentenceProb;
//			if (iS > 0.0 && oS > 0.0 && scalingFactor != 0.0 && d == 0.0)
//			{
//				System.out.println("possible underflow");
//			}
//			assert d <1.01;
			
			tmp[substate] = d;
		}

		return ArrayUtil.logSum(tmp);
		// if (posterior > threshold) {
		// allowedSubStates[start][end][state][substate]=true;
		// nowPossible++;
		// nonePossible=false;
		// } else {
		// allowedSubStates[start][end][state][substate] = false;
		// }

		// if (nonePossible) allowedSubStates[start][end][state] = null;

	}
	
	public void doConstrainedPosteriorInsideOutsideScores(List<String> sentence,
			boolean[][][][] allowed, boolean noSmoothing, Tree<StateSet> goldTree,
			List<String> posTags, double[][][] posteriors) {

		length = (short) sentence.size();
		this.posteriors = posteriors;
		if (allowed != null)
			allowedSubStates = allowed;
		// else setConstraints(null, false);
		// if (boostIncorrect&&goldTree!=null) setGoldProductions(goldTree,false);

		this.goldTree = goldTree;
		createPosteriorArrays();
		scrubPosteriorArrays();
		initializePosteriorChart(sentence, noSmoothing, posTags);

		// initializeChart(sentence, noSmoothing, posTags);

		doConstrainedPosteriorInsideScores();
//		for (int i = 0; i < iPosteriorScorePreU.length; ++i) {
//			if (oPosteriorScorePreU[i] == null)
//				continue;
//			for (int j = 0; j < iPosteriorScorePreU[i].length; ++j) {
//				if (oPosteriorScorePreU[i][j] == null)
//					continue;
//				for (int k = 0; k < iPosteriorScorePreU[i][j].length; ++k) {
//					if (oPosteriorScorePreU[i][j][k] == null)
//						continue;
////					System.out.println(i + "," + j + "," + k + ":"
////							+ oPosteriorScorePreU[i][j][k][0]
////							* Math.exp(ScalingTools.LOGSCALE*oPosteriorScale[i][j][k]) + " with scale " + oPosteriorScale[i][j][k]);
////					System.out.println(i + "," + j + "," + k + ":"
////							+ oPosteriorScorePostU[i][j][k][0]
////							* Math.exp(ScalingTools.LOGSCALE* oPosteriorScale[i][j][k]));
//				}
//			}
//		}

		// if ((10*iScale[0][length][0])!=0)
		// System.out.println("scale "+iScale[0][length][0]);
		// System.out.println("Found a parse for sentence with length "+length+".
		// The LL is "+logLikelihood+".");

		// oScorePreU[0][length][0][0] = 1.0;
		// oScale[0][length][0] = 0;
		doConstrainedPosteriorOutsideScores();
//		System.out.println("-------");
//		for (int i = 0; i < iPosteriorScorePreU.length; ++i) {
//			if (oPosteriorScorePreU[i] == null)
//				continue;
//			for (int j = 0; j < iPosteriorScorePreU[i].length; ++j) {
//				if (oPosteriorScorePreU[i][j] == null)
//					continue;
//				for (int k = 0; k < iPosteriorScorePreU[i][j].length; ++k) {
//					if (oPosteriorScorePreU[i][j][k] == null)
//						continue;
////					System.out.println(i + "," + j + "," + k + ":"
////							+ oPosteriorScorePreU[i][j][k][0]
////							* Math.exp(ScalingTools.LOGSCALE*oPosteriorScale[i][j][k]) + " with scale " + oPosteriorScale[i][j][k]);
////					System.out.println(i + "," + j + "," + k + ":"
////							+ oPosteriorScorePostU[i][j][k][0]
////							* Math.exp(ScalingTools.LOGSCALE*oPosteriorScale[i][j][k]));
//				}
//			}
//		}

//		System.out.println("yeah!");

	}

	private void createPosteriorArrays() {
		if (goldTreeAsSet == null)
		{
			goldTreeAsSet = new HashMap<MyStateSet,Pair<Boolean, Tree<StateSet>>>();
		}
		if (iPosteriorScorePostU == null || iPosteriorScorePostU.length < length) {
			iPosteriorScorePreU = new double[length][length + 1][][];
			iPosteriorScorePostU = new double[length][length + 1][][];
			oPosteriorScorePreU = new double[length][length + 1][][];
			oPosteriorScorePostU = new double[length][length + 1][][];
			outsideAlreadyAdded = new boolean[length][length + 1][];
			iPosteriorScale = new int[length][length + 1][];
			oPosteriorScale = new int[length][length + 1][];
			// insideAlreadyAdded = new boolean[length][length + 1][];

			for (int start = 0; start < length; start++) {
				for (int end = start + 1; end <= length; end++) {
					iPosteriorScorePreU[start][end] = new double[numStates][];
					iPosteriorScorePostU[start][end] = new double[numStates][];
					oPosteriorScorePreU[start][end] = new double[numStates][];
					oPosteriorScorePostU[start][end] = new double[numStates][];
					outsideAlreadyAdded[start][end] = new boolean[numStates];
					iPosteriorScale[start][end] = new int[numStates];
					oPosteriorScale[start][end] = new int[numStates];
					Arrays.fill(iPosteriorScale[start][end], Integer.MIN_VALUE);
					Arrays.fill(oPosteriorScale[start][end], Integer.MIN_VALUE);
					// insideAlreadyAdded[start][end] = new boolean[numStates];

					for (int state = 0; state < numSubStatesArray.length; state++) {
						if (end - start > 1 && !grammarTags[state])
							continue;
						iPosteriorScorePreU[start][end][state] = new double[numSubStatesArray[state]];
						iPosteriorScorePostU[start][end][state] = new double[numSubStatesArray[state]];
						oPosteriorScorePreU[start][end][state] = new double[numSubStatesArray[state]];
						oPosteriorScorePostU[start][end][state] = new double[numSubStatesArray[state]];
					}
				}
			}
		}
	}

	protected void scrubPosteriorArrays() {
		goldTreeAsSet.clear();
		if (iPosteriorScorePostU == null)
			return;
		for (int start = 0; start < length; start++) {
			for (int end = start + 1; end <= length; end++) {
				Arrays.fill(outsideAlreadyAdded[start][end], false);
				Arrays.fill(iPosteriorScale[start][end], Integer.MIN_VALUE);
				Arrays.fill(oPosteriorScale[start][end], Integer.MIN_VALUE);
				// Arrays.fill(insideAlreadyAdded[start][end], false);
				for (int state = 0; state < numSubStatesArray.length; state++) {
					if (end - start > 1 && !grammarTags[state])
						continue;
					if (allowedSubStates[start][end][state] != null) {
						Arrays.fill(iPosteriorScorePreU[start][end][state], 0);
						Arrays.fill(iPosteriorScorePostU[start][end][state], 0);
						Arrays.fill(oPosteriorScorePreU[start][end][state], 0);
						Arrays.fill(oPosteriorScorePostU[start][end][state], 0);

					}
				}
			}
		}
	}

	private static class MyStateSet
	{
		private short state;
		private int from;
		private int to;
		/**
		 * @param state
		 * @param from
		 * @param to
		 */
		public MyStateSet(short state, int from, int to) {
			super();
			this.state = state;
			this.from = from;
			this.to = to;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + from;
			result = prime * result + state;
			result = prime * result + to;
			return result;
		}
		
		@Override
		public String toString()
		{
			return state + "(" + from + "," + to + ")";
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			final MyStateSet other = (MyStateSet) obj;
			if (from != other.from)
				return false;
			if (state != other.state)
				return false;
			if (to != other.to)
				return false;
			return true;
		}
	}
	
	
	
	private Map<MyStateSet,Pair<Boolean, Tree<StateSet>>> goldTreeAsSet = null;
	
	private boolean inTree(Tree<StateSet> tree, int start, int end, int state,
			boolean isUnary, boolean isInside) {
		if (goldTree == null)
		{
			//XXX fix me!
			boolean valid = iScorePostU[start][end][state] != null && oScorePostU[start][end][state] != null && iScorePostU[start][end][state][0] != 0.0 && oScorePostU[start][end][state][0] != 0.0;
			if (!valid) return false;
			final boolean isUnarySymbol = ((String)Numberer.getGlobalNumberer("tags").object(state)).contains("^u");
			if (isUnarySymbol)
				return ((isUnary && isInside) || ((!isUnary && !isInside)));
			else
			{
				return isUnary ^ isInside ;
			}
		}

		int prevFrom = -1;
		int prevTo = -1;
		if (goldTreeAsSet.isEmpty())
		{
			for (Tree<StateSet> nt : tree.getNonTerminals()) {
				StateSet s = nt.getLabel();
				boolean isFirstOnSpan = !(s.from == prevFrom && s.to == prevTo);
				prevTo = s.to;
				prevFrom = s.from;
					
				MyStateSet m = new MyStateSet(s.getState(),s.from,s.to);
				assert (!goldTreeAsSet.containsKey(m));
				
				goldTreeAsSet.put(m,new Pair<Boolean,Tree<StateSet>>(isFirstOnSpan,nt));
			}
		}
		

		final Pair<Boolean,Tree<StateSet>> nonTerminalsOnSpan = goldTreeAsSet.get(new MyStateSet((short)state,start,end));
		if (nonTerminalsOnSpan == null)
			return false;
		Tree<StateSet> nt = nonTerminalsOnSpan.getSecond();
		boolean isFirstOnSpan = nonTerminalsOnSpan.getFirst();
		
					if (nt.getChildren().size() == 1 && !nt.isPreTerminal()) {
						final boolean b = (isInside && isUnary) || (!isInside && !isUnary);
						return b;
					} else {
						if (isFirstOnSpan) {
							return !isUnary;
						} else {
							final boolean b = (isInside && !isUnary)
									|| (!isInside && isUnary);
							return b;
						}

					}

				

		
	}

	void doConstrainedPosteriorInsideScores() {
		double initVal = 0;
		int smallestScale = 10, largestScale = -10;
		for (int diff = 1; diff <= length; diff++) {
			// smallestScale = 10; largestScale = -10;
			// System.out.print(diff + " ");
			for (int start = 0; start < (length - diff + 1); start++) {
				int end = start + diff;
				for (int pState = 0; pState < numSubStatesArray.length; pState++) {
					if (diff == 1)
						continue; // there are no binary rules that span over 1 symbol only
					if (allowedSubStates[start][end][pState] == null)
						continue;
					int nParentStates = numSubStatesArray[pState];
					boolean inTree = inTree(goldTree, start, end, pState, false, true);
					if (inTree) {
						int parentScale = iPosteriorScale[start][end][pState];

						// boolean firstTime = false;
						int currentScale = -oScale[start][end][pState];
						parentScale = parentScale == Integer.MIN_VALUE ? currentScale
								: parentScale;

						double denom = 0.0;
						for (int np = 0; np < nParentStates; ++np) {
							unscaledScoresToAdd1[np] = iScorePostU[start][end][pState][np];
							denom += (iScorePostU[start][end][pState][np] * oScorePostU[start][end][pState][np]);
						}
						assert denom > 0.0;
						ArrayUtil.multiplyInPlace(unscaledScoresToAdd1, 1.0 / denom);
						iPosteriorScale[start][end][pState] = matchArrayScales(
								unscaledScoresToAdd1, currentScale,
								iPosteriorScorePreU[start][end][pState], parentScale);
						assert !ScalingTools.isBadScale(iPosteriorScale[start][end][pState]);
//
//						 System.out.println("Adding inside for " +
//						 edu.berkeley.nlp.util.Numberer.getGlobalNumberer("tags").object(pState)
//						 + " at binary phase for span " + start + " " + end);
						for (int np = 0; np < nParentStates; ++np) {
							iPosteriorScorePreU[start][end][pState][np] += unscaledScoresToAdd1[np];

						}
						Arrays.fill(unscaledScoresToAdd1, initVal);
					}

					BinaryRule[] parentRules = grammar.splitRulesWithP(pState);

					boolean somethingChanged = false;
					for (int i = 0; i < parentRules.length; i++) {
						BinaryRule r = parentRules[i];
						int lState = r.leftChildState;
						int rState = r.rightChildState;

						int narrowR = narrowRExtent[start][lState];
						boolean iPossibleL = (narrowR < end); // can this left constituent
						// leave space for a right
						// constituent?
						if (!iPossibleL) {
							continue;
						}

						int narrowL = narrowLExtent[end][rState];
						boolean iPossibleR = (narrowL >= narrowR); // can this right
						// constituent fit next
						// to the left
						// constituent?
						if (!iPossibleR) {
							continue;
						}

						int min1 = narrowR;
						int min2 = wideLExtent[end][rState];
						int min = (min1 > min2 ? min1 : min2); // can this right
						// constituent stretch far
						// enough to reach the left
						// constituent?
						if (min > narrowL) {
							continue;
						}

						int max1 = wideRExtent[start][lState];
						int max2 = narrowL;
						int max = (max1 < max2 ? max1 : max2); // can this left constituent
						// stretch far enough to
						// reach the right
						// constituent?
						if (min > max) {
							continue;
						}

						// TODO switch order of loops for efficiency
						double[][][] scores = r.getScores2();
						int nLeftChildStates = numSubStatesArray[lState];
						int nRightChildStates = numSubStatesArray[rState];

						for (int split = min; split <= max; split++) {
							boolean changeThisRound = false;
							if (allowedSubStates[start][split][lState] == null)
								continue;
							if (allowedSubStates[split][end][rState] == null)
								continue;

							for (int lp = 0; lp < nLeftChildStates; lp++) {
								double lS1 = iScorePostU[start][split][lState][lp];
								
								double lS2 = iPosteriorScorePostU[start][split][lState][lp];
								if (lS2 == initVal && lS1 == initVal)
									continue;

								for (int rp = 0; rp < nRightChildStates; rp++) {
									if (scores[lp][rp] == null)
										continue;
									double rS1 = iScorePostU[split][end][rState][rp];
									

									double rS2 = iPosteriorScorePostU[split][end][rState][rp];
									if (rS2 == initVal && rS1 == initVal)
										continue;

									for (int np = 0; np < nParentStates; np++) {
										if (!allowedSubStates[start][end][pState][np])
											continue;
										double pS = scores[lp][rp][np];
										if (pS == initVal)
											continue;

										double thisRound1 = pS * (lS1 * rS2);
										double thisRound2 = pS * (lS2 * rS1);
										if (thisRound1 == 0 && thisRound2 == 0)
											continue;
										// if (boostThisRule)
										// thisRound *= boostingFactor;

										unscaledScoresToAdd1[np] += thisRound1;
										unscaledScoresToAdd2[np] += thisRound2;

										somethingChanged = true;
										changeThisRound = true;
									}
								}
							}
							if (!changeThisRound)
								continue;
							// boolean firstTime = false;

							int currentScale1 = addScales(iScale[start][split][lState]
									, iPosteriorScale[split][end][rState]);
							int currentScale2 = addScales(iPosteriorScale[start][split][lState]
									, iScale[split][end][rState]);
							int parentScale = (iPosteriorScale[start][end][pState] == Integer.MIN_VALUE) ? currentScale1
									: iPosteriorScale[start][end][pState];
							currentScale1 = ScalingTools.scaleArray(unscaledScoresToAdd1,
									currentScale1);
							currentScale2 = ScalingTools.scaleArray(unscaledScoresToAdd2,
									currentScale2);
							iPosteriorScale[start][end][pState] = matchArrayScales(
									unscaledScoresToAdd1, currentScale1, unscaledScoresToAdd2,
									currentScale2, iPosteriorScorePreU[start][end][pState],
									parentScale);
							assert !ScalingTools.isBadScale(iPosteriorScale[start][end][pState]);
							// if (parentScale != currentScale) {
							// assert false;
							// if (parentScale == Integer.MIN_VALUE) { // first time to build
							// // this span
							// iScale[start][end][pState] = currentScale;
							// } else {
							// int newScale = Math.max(currentScale, parentScale);
							// ScalingTools.scaleArrayToScale(unscaledScoresToAdd,
							// currentScale, newScale);
							// ScalingTools.scaleArrayToScale(
							// iPosteriorScorePreU[start][end][pState], parentScale,
							// newScale);
							// iScale[start][end][pState] = newScale;
							// }
							// }

							for (int np = 0; np < nParentStates; np++) {

								iPosteriorScorePreU[start][end][pState][np] += unscaledScoresToAdd1[np];
								iPosteriorScorePreU[start][end][pState][np] += unscaledScoresToAdd2[np];

							}

							Arrays.fill(unscaledScoresToAdd1, 0);
							Arrays.fill(unscaledScoresToAdd2, 0);
						}
					}
					// if (somethingChanged) {
					// if (start > narrowLExtent[end][pState]) {
					// narrowLExtent[end][pState] = start;
					// wideLExtent[end][pState] = start;
					// } else {
					// if (start < wideLExtent[end][pState]) {
					// wideLExtent[end][pState] = start;
					// }
					// }
					// if (end < narrowRExtent[start][pState]) {
					// narrowRExtent[start][pState] = end;
					// wideRExtent[start][pState] = end;
					// } else {
					// if (end > wideRExtent[start][pState]) {
					// wideRExtent[start][pState] = end;
					// }
					// }
					// }
				}
				// now do the unaries
				for (int pState = 0; pState < numSubStatesArray.length; pState++) {
					if (allowedSubStates[start][end][pState] == null)
						continue;
					if (iScorePreU[start][end][pState] == null)
						continue;
					// Should be: Closure under sum-product:
					UnaryRule[] unaries = grammar.getClosedSumUnaryRulesByParent(pState);
					// UnaryRule[] unaries =
					grammar.getUnaryRulesByParent(pState).toArray(new UnaryRule[0]);

					int nParentStates = numSubStatesArray[pState];// scores[0].length;

					boolean inTree = inTree(goldTree, start, end, pState, true, true);
					if (inTree) {
						int parentScale = iPosteriorScale[start][end][pState];

						// boolean firstTime = false;
						int currentScale = -oScale[start][end][pState];
						parentScale = parentScale == Integer.MIN_VALUE ? currentScale
								: parentScale;
						double denom = 0.0;
						for (int np = 0; np < nParentStates; ++np) {
							unscaledScoresToAdd1[np] = iScorePostU[start][end][pState][np];
							denom += iScorePostU[start][end][pState][np] * oScorePostU[start][end][pState][np];
						}
						assert denom > 0.0;
						ArrayUtil.multiplyInPlace(unscaledScoresToAdd1, 1.0 / denom);

						iPosteriorScale[start][end][pState] = matchArrayScales(
								unscaledScoresToAdd1, currentScale,
								iPosteriorScorePostU[start][end][pState], parentScale);
						assert !ScalingTools.isBadScale(iPosteriorScale[start][end][pState]);

						for (int np = 0; np < nParentStates; np++) {

							iPosteriorScorePostU[start][end][pState][np] += unscaledScoresToAdd1[np];
//							 System.out.println("Adding inside for " +
//							 edu.berkeley.nlp.util.Numberer.getGlobalNumberer("tags").object(pState)
//							 + " at unary phase for span " + start + " " + end);
						}
						Arrays.fill(unscaledScoresToAdd1, initVal);

					}

					int parentScale = iPosteriorScale[start][end][pState];
					int scaleBeforeUnaries = parentScale;
					boolean somethingChanged = false;

					for (int r = 0; r < unaries.length; r++) {
						UnaryRule ur = unaries[r];
						int cState = ur.childState;
						if ((pState == cState))
							continue;

						if (allowedSubStates[start][end][cState] == null)
							continue;
						if (iScorePreU[start][end][cState] == null)
							continue;

						double[][] scores = ur.getScores2();
						boolean changeThisRound = false;
						int nChildStates = numSubStatesArray[cState];// scores[0].length;
						for (int cp = 0; cp < nChildStates; cp++) {
							if (scores[cp] == null)
								continue;
							double iS = iPosteriorScorePreU[start][end][cState][cp];
							if (iS == initVal)
								continue;

							for (int np = 0; np < nParentStates; np++) {
								if (!allowedSubStates[start][end][pState][np])
									continue;
								double pS = scores[cp][np];
								if (pS == initVal)
									continue;

								double thisRound = iS * pS;

								unscaledScoresToAdd1[np] += thisRound;

								somethingChanged = true;
								changeThisRound = true;
							}
						}
						if (!changeThisRound)
							continue;

						// boolean firstTime = false;
						int currentScale = iPosteriorScale[start][end][cState];
						currentScale = ScalingTools.scaleArray(unscaledScoresToAdd1,
								currentScale);
						parentScale = matchArrayScales(unscaledScoresToAdd1, currentScale,
								iPosteriorScorePostU[start][end][pState], parentScale);
						// if (parentScale != currentScale) {
						// assert false;
						// if (parentScale == Integer.MIN_VALUE) { // first time to build
						// // this span
						// parentScale = currentScale;
						// } else {
						// int newScale = Math.max(currentScale, parentScale);
						// ScalingTools.scaleArrayToScale(unscaledScoresToAdd,
						// currentScale, newScale);
						// ScalingTools.scaleArrayToScale(
						// iPosteriorScorePostU[start][end][pState], parentScale,
						// newScale);
						// parentScale = newScale;
						// }
						// }

						for (int np = 0; np < nParentStates; np++) {

							iPosteriorScorePostU[start][end][pState][np] += unscaledScoresToAdd1[np];

						}

						// insideAlreadyAdded[start][end][pState] |= inTree;
						Arrays.fill(unscaledScoresToAdd1, 0);
					}
					if (somethingChanged) {

						iPosteriorScale[start][end][pState] = matchArrayScales(
								iPosteriorScorePostU[start][end][pState], parentScale,
								iPosteriorScorePreU[start][end][pState], scaleBeforeUnaries);
						assert !ScalingTools.isBadScale(iPosteriorScale[start][end][pState]) ;
						// int newScale = Math.max(scaleBeforeUnaries, parentScale);
						// ScalingTools.scaleArrayToScale(
						// iPosteriorScorePreU[start][end][pState], scaleBeforeUnaries,
						// newScale);
						// ScalingTools.scaleArrayToScale(
						// iPosteriorScorePostU[start][end][pState], parentScale,
						// newScale);
						// iPosteriorScale[start][end][pState] = newScale;

						// if (start > narrowLExtent[end][pState]) {
						// narrowLExtent[end][pState] = start;
						// wideLExtent[end][pState] = start;
						// } else {
						// if (start < wideLExtent[end][pState]) {
						// wideLExtent[end][pState] = start;
						// }
						// }
						// if (end < narrowRExtent[start][pState]) {
						// narrowRExtent[start][pState] = end;
						// wideRExtent[start][pState] = end;
						// } else {
						// if (end > wideRExtent[start][pState]) {
						// wideRExtent[start][pState] = end;
						// }
						// }
					}
					// in any case copy/add the scores from before
					for (int np = 0; np < nParentStates; np++) {
						double val = iPosteriorScorePreU[start][end][pState][np];
						if (val > 0) {

							iPosteriorScorePostU[start][end][pState][np] += val;

						}
					}
				}
			}
		}
	}

	void doConstrainedPosteriorOutsideScores() {
		double initVal = 0;

		// Arrays.fill(scoresToAdd,initVal);
		for (int diff = length; diff >= 1; diff--) {
			for (int start = 0; start + diff <= length; start++) {
				int end = start + diff;
				// do unaries
				
				for (int cState = 0; cState < numSubStatesArray.length; cState++) {
					if (allowedSubStates[start][end][cState] == null)
						continue;
					if (iScorePostU[start][end][cState] == null)
						continue;
					if (end - start > 1 && !grammarTags[cState])
						continue;
					int childScale = oPosteriorScale[start][end][cState];
					int scaleBeforeUnaries = childScale;
					// Should be: Closure under sum-product:
					// UnaryRule[] rules =
					// grammar.getClosedSumUnaryRulesByParent(pState);
					UnaryRule[] rules = grammar.getClosedSumUnaryRulesByChild(cState);
					// UnaryRule[] rules =
					// grammar.getClosedViterbiUnaryRulesByParent(pState);
					// For now:
					// UnaryRule[] rules =
					// grammar.getUnaryRulesByChild(cState).toArray(new UnaryRule[0]);
					int nChildStates = numSubStatesArray[cState];

					boolean inTree = inTree(goldTree, start, end, cState, true, false);

					if (inTree) {
						

						// boolean firstTime = false;
						int currentScale = -iScale[start][end][cState];
//						parentScale = parentScale == Integer.MIN_VALUE ? currentScale
//								: parentScale;
						double denom = 0.0;
						for (int cp = 0; cp < nChildStates; ++cp) {
							final double d = oScorePostU[start][end][cState][cp];
							denom += oScorePostU[start][end][cState][cp] * iScorePostU[start][end][cState][cp];
							assert !SloppyMath.isVeryDangerous(d);
							unscaledScoresToAdd1[cp] = d;
						
						}
						assert denom > 0.0;
						ArrayUtil.multiplyInPlace(unscaledScoresToAdd1, 1.0 / denom);

						childScale = matchArrayScales(
								unscaledScoresToAdd1, currentScale,
								oPosteriorScorePostU[start][end][cState], childScale);

						for (int cp = 0; cp < nChildStates; cp++) {

							oPosteriorScorePostU[start][end][cState][cp] += unscaledScoresToAdd1[cp];
//							 System.out.println("Adding outside for " +
//							 edu.berkeley.nlp.util.Numberer.getGlobalNumberer("tags").object(cState)
//							 + " at unary phase for span " + start + " " + end);

						}
						Arrays.fill(unscaledScoresToAdd1, initVal);
					}
					boolean somethingChanged = false;
					
					for (int r = 0; r < rules.length; r++) {
						UnaryRule ur = rules[r];
						int pState = ur.parentState;
						if ((pState == cState))
							continue;
						if (allowedSubStates[start][end][pState] == null)
							continue;
						if (iScorePostU[start][end][pState] == null)
							continue;

						int nParentStates = numSubStatesArray[pState];

						double[][] scores = ur.getScores2();
						boolean changeThisRound = false;
						for (int cp = 0; cp < nChildStates; cp++) {
							if (scores[cp] == null)
								continue;
							if (!allowedSubStates[start][end][cState][cp])
								continue;
							for (int np = 0; np < nParentStates; np++) {
								if (!allowedSubStates[start][end][pState][np])
									continue;
								double pS = scores[cp][np];
								if (pS == initVal)
									continue;

								double oS = oPosteriorScorePreU[start][end][pState][np];
								if (oS == initVal)
									continue;

								double thisRound = oS * pS;
								if (thisRound == 0)
									continue;

								assert !SloppyMath.isVeryDangerous(thisRound);
								unscaledScoresToAdd1[cp] += thisRound;

								somethingChanged = true;
								changeThisRound = true;
							}
						}
						if (!changeThisRound)
							continue;

						int currentScale = oPosteriorScale[start][end][pState];
						currentScale = ScalingTools.scaleArray(unscaledScoresToAdd1,
								currentScale);
						childScale = matchArrayScales(unscaledScoresToAdd1, currentScale,
								oPosteriorScorePostU[start][end][cState], childScale);
						// if (childScale != currentScale) {
						// assert false;
						// if (childScale == Integer.MIN_VALUE) { // first time to build
						// // this span
						// childScale = currentScale;
						// } else {
						// int newScale = Math.max(currentScale, childScale);
						// ScalingTools.scaleArrayToScale(unscaledScoresToAdd1,
						// currentScale, newScale);
						// ScalingTools.scaleArrayToScale(
						// oPosteriorScorePostU[start][end][cState], childScale,
						// newScale);
						// childScale = newScale;
						// }
						// }

						for (int cp = 0; cp < nChildStates; cp++) {

							oPosteriorScorePostU[start][end][cState][cp] += unscaledScoresToAdd1[cp];

						}

						Arrays.fill(unscaledScoresToAdd1, initVal);
					}
					if (somethingChanged) {
						oPosteriorScale[start][end][cState] = matchArrayScales(
								oPosteriorScorePreU[start][end][cState], scaleBeforeUnaries,
								oPosteriorScorePostU[start][end][cState], childScale);
						assert !ScalingTools.isBadScale(oPosteriorScale[start][end][cState]) ;
						// assert scaleBeforeUnaries == childScale;
						// if (false) {
						// int newScale = Math.max(scaleBeforeUnaries, childScale);
						// ScalingTools.scaleArrayToScale(
						// oPosteriorScorePreU[start][end][cState], scaleBeforeUnaries,
						// newScale);
						// ScalingTools.scaleArrayToScale(
						// oPosteriorScorePostU[start][end][cState], childScale,
						// newScale);
						// oScale[start][end][cState] = newScale;
						// }
					}
					// copy/add the entries where the unaries where not useful
					for (int cp = 0; cp < nChildStates; cp++) {
						double val = oPosteriorScorePreU[start][end][cState][cp];
						if (val > 0) {

							oPosteriorScorePostU[start][end][cState][cp] += val;

						}
					}
				}

				// do binaries
				if (diff == 1)
					continue; // there is no space for a binary
				for (int pState = 0; pState < numSubStatesArray.length; pState++) {
					if (allowedSubStates[start][end][pState] == null)
						continue;
					final int nParentChildStates = numSubStatesArray[pState];
					BinaryRule[] rules = grammar.splitRulesWithP(pState);

					// BinaryRule[] rules = grammar.splitRulesWithLC(lState);
					for (int r = 0; r < rules.length; r++) {
						BinaryRule br = rules[r];
						int lState = br.leftChildState;
						int min1 = narrowRExtent[start][lState];
						if (end < min1) {
							continue;
						}

						int rState = br.rightChildState;
						int max1 = narrowLExtent[end][rState];
						if (max1 < min1) {
							continue;
						}

						int min = min1;
						int max = max1;
						if (max - min > 2) {
							int min2 = wideLExtent[end][rState];
							min = (min1 > min2 ? min1 : min2);
							if (max1 < min) {
								continue;
							}
							int max2 = wideRExtent[start][lState];
							max = (max1 < max2 ? max1 : max2);
							if (max < min) {
								continue;
							}
						}

						double[][][] scores = br.getScores2();
						int nLeftChildStates = numSubStatesArray[lState];
						int nRightChildStates = numSubStatesArray[rState];

						for (int split = min; split <= max; split++) {
							if (allowedSubStates[start][split][lState] == null)
								continue;
							if (allowedSubStates[split][end][rState] == null)
								continue;
							if (split - start > 1 && !grammarTags[lState])
								continue;
							if (end - split > 1 && !grammarTags[rState])
								continue;

							boolean inTreeL = !outsideAlreadyAdded[start][split][lState]
									&& inTree(goldTree, start, split, lState, false, false);
							boolean inTreeR = !outsideAlreadyAdded[split][end][rState]
									&& inTree(goldTree, split, end, rState, false, false);
							if (inTreeL) {
								int parentScale = oPosteriorScale[start][split][lState];

								// boolean firstTime = false;
								int currentScale = -iScale[start][split][lState];
								parentScale = parentScale == Integer.MIN_VALUE ? currentScale
										: parentScale;
								double denom = 0.0;
								for (int cp = 0; cp < nLeftChildStates; ++cp) {
									final double d = (oScorePostU[start][split][lState][cp]);
									denom += oScorePostU[start][split][lState][cp] * iScorePostU[start][split][lState][cp];
									assert !SloppyMath.isVeryDangerous(d);
									unscaledScoresToAdd1[cp] = d;
								}
								assert denom > 0.0;
								ArrayUtil.multiplyInPlace(unscaledScoresToAdd1, 1.0 / denom);

								oPosteriorScale[start][split][lState] = matchArrayScales(
										unscaledScoresToAdd1, currentScale,
										oPosteriorScorePreU[start][split][lState], parentScale);
								assert !ScalingTools.isBadScale(oPosteriorScale[start][split][lState]);

								for (int cp = 0; cp < nLeftChildStates; cp++) {

									oPosteriorScorePreU[start][split][lState][cp] += unscaledScoresToAdd1[cp];
//									 System.out.println("Adding outside for " +
//									 edu.berkeley.nlp.util.Numberer.getGlobalNumberer("tags").object(lState)
//									 + " at binary phase for span "
//									 + start
//									 + " " +
//									 split);

								}
								Arrays.fill(unscaledScoresToAdd1, initVal);
							}
							outsideAlreadyAdded[start][split][lState] |= inTreeL;
							outsideAlreadyAdded[split][end][rState] |= inTreeR;

							if (inTreeR) {
								int parentScale = oPosteriorScale[split][end][rState];

								// boolean firstTime = false;
								int currentScale = -iScale[split][end][rState];
								parentScale = parentScale == Integer.MIN_VALUE ? currentScale
										: parentScale;

								double denom = 0.0;
								for (int cp = 0; cp < nLeftChildStates; ++cp) {
									unscaledScoresToAdd1[cp] = (oScorePostU[split][end][rState][cp]);
									denom += iScorePostU[split][end][rState][cp] * oScorePostU[split][end][rState][cp];
								}

								assert denom > 0.0;
								ArrayUtil.multiplyInPlace(unscaledScoresToAdd1, 1.0 / denom);
								oPosteriorScale[split][end][rState] = matchArrayScales(
										unscaledScoresToAdd1, currentScale,
										oPosteriorScorePreU[split][end][rState], parentScale);
								assert !ScalingTools.isBadScale(oPosteriorScale[split][end][rState]) ;
								for (int cp = 0; cp < nRightChildStates; cp++) {

									oPosteriorScorePreU[split][end][rState][cp] += unscaledScoresToAdd1[cp];
//									 System.out.println("Adding outside for " +
//									 edu.berkeley.nlp.util.Numberer.getGlobalNumberer("tags").object(rState)
//									 + " at binary phase for span " + split + " " + end);

								}
								Arrays.fill(unscaledScoresToAdd1, initVal);
							}


							boolean somethingChanged = false;
							for (int lp = 0; lp < nLeftChildStates; lp++) {
								double lS1 = iScorePostU[start][split][lState][lp];
								double lS2 = iPosteriorScorePostU[start][split][lState][lp];
								// if (lS==0) continue;

								for (int rp = 0; rp < nRightChildStates; rp++) {
									if (scores[lp][rp] == null)
										continue;
									double rS1 = iScorePostU[split][end][rState][rp];
									double rS2 = iPosteriorScorePostU[split][end][rState][rp];
									// if (rS==0) continue;

									for (int np = 0; np < nParentChildStates; np++) {
										double pS = scores[lp][rp][np];
										if (pS == initVal)
											continue;

										double oS1 = oScorePostU[start][end][pState][np];
									

										double oS2 = oPosteriorScorePostU[start][end][pState][np];
										if (oS2 == initVal && oS1 == initVal)
											continue;
										// if (!allowedSubStates[start][end][pState][np]) continue;

										double thisRoundL1 = pS * (rS1 * oS2);
										double thisRoundR1 = pS * (lS1 * oS2);
										double thisRoundL2 = pS * (rS2 * oS1);
										double thisRoundR2 = pS * (lS2 * oS1);
										if (thisRoundL1 == 0 && thisRoundR1 == 0 && thisRoundR2 == 0 && thisRoundL2 == 0)
											continue;

										// if (boostThisRule) {
										// thisRoundL *= boostingFactor;
										// thisRoundR *= boostingFactor;
										// }

										scoresToAdd1[lp] += thisRoundL1;
										unscaledScoresToAdd1[rp] += thisRoundR1;
										scoresToAdd2[lp] += thisRoundL2;
										unscaledScoresToAdd2[rp] += thisRoundR2;
										assert !SloppyMath.isVeryDangerous(thisRoundR2);
										assert !SloppyMath.isVeryDangerous(thisRoundL2);
										assert !SloppyMath.isVeryDangerous(thisRoundR1);
										assert !SloppyMath.isVeryDangerous(thisRoundL1);

										somethingChanged = true;
									}
								}
							}
							if (!somethingChanged)
								continue;

							if (ArrayUtil.max(scoresToAdd1) != 0 ||ArrayUtil.max(scoresToAdd2) != 0  ) {// oScale[start][end][pState]!=Integer.MIN_VALUE
								// &&
								// iScale[split][end][rState]!=Integer.MIN_VALUE){
								int leftScale = oPosteriorScale[start][split][lState];
								int currentScale1 = addScales(oPosteriorScale[start][end][pState]
										, iScale[split][end][rState]);
								currentScale1 = ScalingTools.scaleArray(scoresToAdd1,
										currentScale1);
								int currentScale2 = addScales(oScale[start][end][pState]
										,iPosteriorScale[split][end][rState]);
								currentScale2 = ScalingTools.scaleArray(scoresToAdd2,
										currentScale2);
								oPosteriorScale[start][split][lState] = matchArrayScales(
										scoresToAdd1, currentScale1, scoresToAdd2, currentScale2,
										oPosteriorScorePreU[start][split][lState], leftScale);
								assert !ScalingTools.isBadScale(oPosteriorScale[start][split][lState]) ;
								// if (leftScale != currentScale) {
								// assert false;
								// if (leftScale == Integer.MIN_VALUE) { // first time to build
								// // this span
								// oScale[start][split][lState] = currentScale;
								// } else {
								// int newScale = Math.max(currentScale, leftScale);
								// ScalingTools.scaleArrayToScale(scoresToAdd, currentScale,
								// newScale);
								// ScalingTools.scaleArrayToScale(
								// oPosteriorScorePreU[start][split][lState], leftScale,
								// newScale);
								// oScale[start][split][lState] = newScale;
								// }
								// }

								for (int cp = 0; cp < nLeftChildStates; cp++) {
									if (scoresToAdd1[cp] > initVal || scoresToAdd2[cp] > initVal) {

										oPosteriorScorePreU[start][split][lState][cp] += scoresToAdd1[cp];
										oPosteriorScorePreU[start][split][lState][cp] += scoresToAdd2[cp];

									}
								}

								Arrays.fill(scoresToAdd1, 0);
								Arrays.fill(scoresToAdd2, 0);
							}

							if (ArrayUtil.max(unscaledScoresToAdd1) != 0 || ArrayUtil.max(unscaledScoresToAdd2) != 0) {// oScale[start][end][pState]!=Integer.MIN_VALUE
								// &&
								// iScale[start][split][lState]!=Integer.MIN_VALUE){
								int rightScale = oPosteriorScale[split][end][rState];
								int currentScale1 = addScales(oPosteriorScale[start][end][pState]
										, iScale[start][split][lState]);
								
							
							
								int currentScale2 = addScales(oScale[start][end][pState],
										 iPosteriorScale[start][split][lState]);
//								System.out.println(currentScale1 + ":1:" + currentScale2 + "::" + rightScale);
//								System.out.println(unscaledScoresToAdd1[0] + "&&" + unscaledScoresToAdd2[0]);
								currentScale1 = ScalingTools.scaleArray(unscaledScoresToAdd1,
										currentScale1);
								currentScale2 = ScalingTools.scaleArray(unscaledScoresToAdd2,
										currentScale2);
//								System.out.println(currentScale1 + ":2:" + currentScale2 + "::" + rightScale);
//								System.out.println(unscaledScoresToAdd1[0] + "&&" + unscaledScoresToAdd2[0]);
								oPosteriorScale[split][end][rState] = matchArrayScales(
										unscaledScoresToAdd1, currentScale1, unscaledScoresToAdd2, currentScale2,
										oPosteriorScorePreU[split][end][rState], rightScale);
								assert !ScalingTools.isBadScale(oPosteriorScale[split][end][rState]) ;
								// if (rightScale != currentScale) {
								// assert false;
								// if (rightScale == Integer.MIN_VALUE) { // first time to build
								// // this span
								// oScale[split][end][rState] = currentScale;
								// } else {
								// int newScale = Math.max(currentScale, rightScale);
								// ScalingTools.scaleArrayToScale(unscaledScoresToAdd,
								// currentScale, newScale);
								// ScalingTools.scaleArrayToScale(
								// oPosteriorScorePreU[split][end][rState], rightScale,
								// newScale);
								// oScale[split][end][rState] = newScale;
								// }
								// }

								for (int cp = 0; cp < nRightChildStates; cp++) {
									if (unscaledScoresToAdd1[cp]  > initVal || unscaledScoresToAdd2[cp]  > initVal) {

										oPosteriorScorePreU[split][end][rState][cp] += unscaledScoresToAdd1[cp];
										oPosteriorScorePreU[split][end][rState][cp] += unscaledScoresToAdd2[cp];

									}
								}

								Arrays.fill(unscaledScoresToAdd1, 0);
								Arrays.fill(unscaledScoresToAdd2, 0);
							}
						}
					}
				}
			}
		}
	}

	/**
	 * @param i
	 * @param j
	 * @return
	 */
	private static int addScales(int i, int j) {
		
		if (i == Integer.MIN_VALUE || j == Integer.MIN_VALUE)
		{
			return Integer.MIN_VALUE;
		}
		return i+j;
	}
	
private static int addScales(int i, int j, int k) {
		
		if (i == Integer.MIN_VALUE || j == Integer.MIN_VALUE || k == Integer.MIN_VALUE)
		{
			return Integer.MIN_VALUE;
		}
		return i+j+k;
	}

	public void incrementExpectedPosteriorGoldCounts(Linearizer linearizer,
			double[] probs, /* , Grammar grammar, Lexicon lexicon, */
			List<StateSet> sentence, boolean hardCounts) {
		// numSubStatesArray = grammar.numSubStates;
		// double tree_score = iScorePostU[0][length][0][0];
		double tree_score = 1.0;
		int tree_scale = iScale[0][length][0];
		if (SloppyMath.isDangerous(tree_score)) {
			System.out
					.println("Training tree has zero probability - presumably underflow!");
			System.exit(-1);
		}

		for (int start = 0; start < length; start++) {
			final int lastState = numSubStatesArray.length;
			StateSet state = sentence.get(start);
			String word = state.getWord();

			for (int tag = 0; tag < lastState; tag++) {
				if (grammar.isGrammarTag(tag))
					continue;
				if (allowedSubStates[start][start + 1][tag] == null)
					continue;
				double scalingFactor = ScalingTools
						.calcScaleFactor(addScales(oPosteriorScale[start][start + 1][tag]
								,iScale[start][start + 1][tag]));
				if (scalingFactor == 0) {
					continue;
				}
//				int startIndexWord = linearizer.getLinearIndex(state.wordIndex, tag);
//				if (startIndexWord == -1)
//					continue;
//				startIndexWord += lexiconOffset;
				final int nSubStates = numSubStatesArray[tag];
				for (short substate = 0; substate < nSubStates; substate++) {
					// weight by the probability of seeing the tag and word together,
					// given the sentence
					double iS1 = iScorePreU[start][start + 1][tag][substate];
					if (iS1 == 0)
						continue;
//					double oS1 = oScorePostU[start][start + 1][tag][substate];
//					if (oS1 == 0)
//						continue;
//					double iS2 = iPosteriorScorePreU[start][start + 1][tag][substate];
//					if (iS2 == 0)
//						continue;
					double oS2 = oPosteriorScorePostU[start][start + 1][tag][substate];
					if (oS2 == 0)
						continue;
					double weight1 = iS1 / tree_score * scalingFactor * oS2;
					// double weight2 = iS2 / tree_score * scalingFactor * oS1;
					assert !SloppyMath.isVeryDangerous(weight1);
					if (weight1 > 1e9)
					{
						System.out.println("Oy lexicon");
					}
					if (weight1 > 0)
//						probs[startIndexWord + substate] += weight1;
						tmpCountsArray[substate] = weight1;
					// if (weight2 > 0)
					// probs[startIndexWord + substate] += weight2;
					// if (weight>1.02)
					// System.out.println("too big");
				}
        linearizer.increment(probs, state, tag, tmpCountsArray, false); //probs[startIndexWord+substate] += weight;
			}
		}

		for (int diff = 1; diff <= length; diff++) {
			for (int start = 0; start < (length - diff + 1); start++) {
				int end = start + diff;

				final int lastState = numSubStatesArray.length;
				for (short pState = 0; pState < lastState; pState++) {
					if (diff == 1)
						continue; // there are no binary rules that span over 1 symbol only
					if (allowedSubStates[start][end][pState] == null)
						continue;
					final int nParentSubStates = numSubStatesArray[pState];
					BinaryRule[] parentRules = grammar.splitRulesWithP(pState);
					for (int i = 0; i < parentRules.length; i++) {
						BinaryRule r = parentRules[i];
						short lState = r.leftChildState;
						short rState = r.rightChildState;

						int narrowR = narrowRExtent[start][lState];
						boolean iPossibleL = (narrowR < end); // can this left constituent
						// leave space for a right
						// constituent?
						if (!iPossibleL) {
							continue;
						}

						int narrowL = narrowLExtent[end][rState];
						boolean iPossibleR = (narrowL >= narrowR); // can this right
						// constituent fit next
						// to the left
						// constituent?
						if (!iPossibleR) {
							continue;
						}

						int min1 = narrowR;
						int min2 = wideLExtent[end][rState];
						int min = (min1 > min2 ? min1 : min2); // can this right
						// constituent stretch far
						// enough to reach the left
						// constituent?
						if (min > narrowL) {
							continue;
						}

						int max1 = wideRExtent[start][lState];
						int max2 = narrowL;
						int max = (max1 < max2 ? max1 : max2); // can this left constituent
						// stretch far enough to
						// reach the right
						// constituent?
						if (min > max) {
							continue;
						}

						double[][][] scores = r.getScores2();
						boolean foundSomething = false;

						for (int split = min; split <= max; split++) {
							if (allowedSubStates[start][split][lState] == null)
								continue;
							if (allowedSubStates[split][end][rState] == null)
								continue;
							double scalingFactor1 = ScalingTools
									.calcScaleFactor(addScales(oPosteriorScale[start][end][pState]
											, iScale[start][split][lState]
											, iScale[split][end][rState]));
							double scalingFactor2 = ScalingTools
							.calcScaleFactor(addScales(oScale[start][end][pState]
									, iPosteriorScale[start][split][lState]
									, iScale[split][end][rState]));
							double scalingFactor3 = ScalingTools
							.calcScaleFactor(addScales(oScale[start][end][pState]
									, iScale[start][split][lState]
									,iPosteriorScale[split][end][rState]));
//							if (scalingFactor == 0) {
//								continue;
//							}

							int curInd = 0;
							for (int lp = 0; lp < scores.length; lp++) {
								double lcIS1 = iScorePostU[start][split][lState][lp];
								double lcIS2 = iPosteriorScorePostU[start][split][lState][lp];
								double tmpA1 = lcIS1 / tree_score;
								double tmpA2 = lcIS2 / tree_score;

								for (int rp = 0; rp < scores[0].length; rp++) {
									if (scores[lp][rp] == null)
										continue;
									double rcIS1 = iScorePostU[split][end][rState][rp];
									double rcIS2 = iPosteriorScorePostU[split][end][rState][rp];
									double tmpB = tmpA1 * rcIS1 * scalingFactor1;
									double tmpB1 = tmpA1 * rcIS2 * scalingFactor3;
									double tmpB2 = tmpA2 * rcIS1 * scalingFactor2;

									for (int np = 0; np < nParentSubStates; np++) {
										curInd++;
										if (lcIS1 == 0 && lcIS2 == 0) {
											continue;
										}
										if (rcIS1 == 0 && rcIS2 == 0) {
											continue;
										}
										double pOS1 = oScorePostU[start][end][pState][np];
										double pOS2 = oPosteriorScorePostU[start][end][pState][np];
										if (pOS1 == 0 && pOS2 == 0) {
											continue;
										}

										double rS = scores[lp][rp][np];
										if (rS == 0) {
											continue;
										}
										double ruleCount1 = (hardCounts) ? 1 : rS * tmpB1 * pOS1;
										double ruleCount2 = (hardCounts) ? 1 : rS * tmpB2 * pOS1;
										double ruleCount3 = (hardCounts) ? 1 : rS * tmpB * pOS2;
										if (ruleCount1 == 0 && ruleCount2 == 0 && ruleCount3 == 0)
											continue;
										if (ruleCount1 + ruleCount2
												+ ruleCount3 > 1e9)
										{
											System.out.println("Oy binary");
										}
										tmpCountsArray[curInd - 1] += ruleCount1 + ruleCount2
												+ ruleCount3;
										foundSomething = true;
									}
								}
							}
						}
						if (!foundSomething)
							continue; // nothing changed this round
//						int thisStartIndex = linearizer.getLinearIndex(new BinaryRule(
//								pState, lState, rState));
//						int curInd = 0;
//						for (int lp = 0; lp < scores.length; lp++) {
//							for (int rp = 0; rp < scores[0].length; rp++) {
//								if (scores[lp][rp] == null)
//									continue;
//								for (int np = 0; np < nParentSubStates; np++) {
//									curInd++;
//									double ruleCount = tmpCountsArray[curInd - 1];
//									assert !SloppyMath.isVeryDangerous(ruleCount);
//									if (ruleCount > 0)
//										probs[thisStartIndex + curInd - 1] += ruleCount;
//										
//								}
//							}
//						}
//						Arrays.fill(tmpCountsArray, 0);
						linearizer.increment(probs, r, tmpCountsArray, false); //probs[thisStartIndex + curInd-1] += ruleCount;
					}
				}
				final int lastStateU = numSubStatesArray.length;
				for (short pState = 0; pState < lastStateU; pState++) {
					if (allowedSubStates[start][end][pState] == null)
						continue;

					// List<UnaryRule> unaries = grammar.getUnaryRulesByParent(pState);
					int nParentSubStates = numSubStatesArray[pState];
					UnaryRule[] unaries = grammar.getClosedSumUnaryRulesByParent(pState);
					for (UnaryRule ur : unaries) {
						short cState = ur.childState;
						if ((pState == cState))
							continue;// && (np == cp))continue;
						if (allowedSubStates[start][end][cState] == null)
							continue;
						double scalingFactor1 = ScalingTools
								.calcScaleFactor(addScales(oPosteriorScale[start][end][pState]
										, iScale[start][end][cState]));
						double scalingFactor2 = ScalingTools
						.calcScaleFactor(addScales(oScale[start][end][pState]
								, iPosteriorScale[start][end][cState]));
//						if (scalingFactor == 0) {
//							continue;
//						}

						double[][] scores = ur.getScores2();
//						int thisStartIndex = linearizer.getLinearIndex(new UnaryRule(
//								pState, cState));
						int curInd = 0;
						for (int cp = 0; cp < scores.length; cp++) {
							if (scores[cp] == null)
								continue;
							double cIS1 = iScorePreU[start][end][cState][cp];
							double cIS2 = iPosteriorScorePreU[start][end][cState][cp];
							double tmpA1 = cIS1 / tree_score * scalingFactor1;
							double tmpA2 = cIS2 / tree_score * scalingFactor2;

							for (int np = 0; np < nParentSubStates; np++) {
								curInd++;
								if (cIS1 == 0 && cIS2 == 0) {
									continue;
								}

								double pOS1 = oScorePreU[start][end][pState][np];
								double pOS2 = oPosteriorScorePreU[start][end][pState][np];
								if (pOS1 == 0 && pOS2 == 0) {
									continue;
								}

								double rS = scores[cp][np];
								if (rS == 0) {
									continue;
								}

								double ruleCount1 = (hardCounts) ? 1 : rS * tmpA1 * pOS2;
								double ruleCount2 = (hardCounts) ? 1 : rS * tmpA2 * pOS1;
								
								if (ruleCount1 == 0 && ruleCount2 == 0)
									continue;
								if (ruleCount1 + ruleCount2 > 1e9)
								{
									System.out.println("Oy unary");
								}
								assert !SloppyMath.isVeryDangerous(ruleCount1+ruleCount2);
//								probs[thisStartIndex + curInd - 1] += ruleCount1 + ruleCount2;
								tmpCountsArray[curInd - 1] += ruleCount1 + ruleCount2;
							}
						}
						linearizer.increment(probs, ur, tmpCountsArray, false); //probs[thisStartIndex + curInd-1] += ruleCount;
					}
				}
			}
		}
	}

	/**
	 * @param linearizer
	 * @param probs
	 * @param sentence
	 * @param hardCounts
	 * @param lexiconOffset
	 */
	public void incrementExpectedPosteriorGoldCountsByEnumeration(
			Linearizer linearizer, double[] probs, List<String> sentence,
			boolean hardCounts, boolean useGoldTree) {
		List<Tree<MyRule>> allTrees = getAllTrees(sentence, (short) 0, 0, sentence
				.size());
		double Z = 0.0;
		for (Tree<MyRule> t : allTrees) {
			final double treeScore = treeScore(t);
			Z += treeScore;
		}
		Set<String> doneWords = new HashSet<String>();
		for (int start = 0; start < length; start++) {
		
			final int lastState = numSubStatesArray.length;
			String word = sentence.get(start);

			for (int tag = 0; tag < lastState; tag++) {
				final String string = sentence.get(start) + "::+::" + tag;
				if (doneWords.contains(string))
					continue;
				
				if (grammar.isGrammarTag(tag))
					continue;
				if (allowedSubStates[start][start + 1][tag] == null)
					continue;

				int startIndexWord = ((DefaultLinearizer)linearizer).getLinearIndex(word, tag);
				if (startIndexWord == -1)
					continue;
//				startIndexWord += lexiconOffset;
				final int nSubStates = numSubStatesArray[tag];
				doneWords.add(string);
				for (short substate = 0; substate < nSubStates; substate++) {
					double weight = getCount(new MyRule(-1, (short) tag, sentence
							.get(start), -1, -1), allTrees, Z,useGoldTree);
					if (weight > 0)
						probs[startIndexWord + substate] += weight;
//						tmpCountsArray[substate] = weight;

				}
				// NOT SURE what to here... it should look like:
//        linearizer.increment(probs, new StateSet(), tag, tmpCountsArray); //probs[startIndexWord+substate] += weight;
			}
		}

		final int lastState = numSubStatesArray.length;
		for (short pState = 0; pState < lastState; pState++) {

			final int nParentSubStates = numSubStatesArray[pState];
			BinaryRule[] parentRules = grammar.splitRulesWithP(pState);
			for (int i = 0; i < parentRules.length; i++) {
				BinaryRule r = parentRules[i];
				short lState = r.leftChildState;
				short rState = r.rightChildState;

//				int thisStartIndex = linearizer.getLinearIndex(new BinaryRule(pState,
//						lState, rState));
				int curInd = 0;
				for (int lp = 0; lp < 1; lp++) {
					for (int rp = 0; rp < 1; rp++) {
						// if (scores[lp][rp] == null)
						// continue;
						for (int np = 0; np < nParentSubStates; np++) {
							curInd++;
							double ruleCount = getCount(new MyRule(-11, r, -1, -1), allTrees,
									Z,useGoldTree);
							if (ruleCount > 0)
//								probs[thisStartIndex + curInd - 1] += ruleCount;
								tmpCountsArray[curInd - 1] += ruleCount;
						}
					}
				}
//				Arrays.fill(tmpCountsArray, 0);
				linearizer.increment(probs, r, tmpCountsArray, true); //probs[thisStartIndex + curInd-1] += ruleCount;
			}
		}
		final int lastStateU = numSubStatesArray.length;
		for (short pState = 0; pState < lastStateU; pState++) {

			// List<UnaryRule> unaries = grammar.getUnaryRulesByParent(pState);
			int nParentSubStates = numSubStatesArray[pState];
			UnaryRule[] unaries = grammar.getClosedSumUnaryRulesByParent(pState);
			for (UnaryRule ur : unaries) {
				short cState = ur.childState;

				double[][] scores = ur.getScores2();
//				int thisStartIndex = linearizer.getLinearIndex(new UnaryRule(pState,
//						cState));
				int curInd = 0;
				for (int cp = 0; cp < scores.length; cp++) {
					if (scores[cp] == null)
						continue;

					for (int np = 0; np < nParentSubStates; np++) {
						curInd++;

//						probs[thisStartIndex + curInd - 1] += getCount(new MyRule(-1, ur,
//								-1, -1), allTrees, Z);
						tmpCountsArray[curInd - 1] += getCount(new MyRule(-1, ur, -1, -1), allTrees, Z,useGoldTree);
					}
				}
				linearizer.increment(probs, ur, tmpCountsArray, true); //probs[thisStartIndex + curInd-1] += ruleCount;
			}
		}

	}

	private double getCount(MyRule rule, List<Tree<MyRule>> allTrees, double Z, boolean useGoldTree) {
		if (useGoldTree)
		{
			
		
		double totalCounts = 0.0;
		final List<Tree<StateSet>> nonTerminals = useGoldTree ?  goldTree.getNonTerminals() : genAllNonTerminals(allTrees);
		for (Tree<StateSet> goldNode : nonTerminals) {
			double norm = 0.0;
			double counts = 0.0;
			for (Tree<MyRule> tree : allTrees) {
				if (hasNode(tree, goldNode)) {
					double score = treeScore(tree);
					int count = count(tree, rule);
					counts += score * count / Z;
					norm += score / Z;

				}
			}
			// System.out.println("enumeration posterior for " + goldNode + " is "
			// + norm);
			if (norm != 0.0)
				totalCounts += counts / norm;
		}
		return totalCounts;
		}
		else
		{
//			Counter<Tree<MyRule>> counts = new Counter<Tree<MyRule>>();
//			Counter<Tree<MyRule>> norm = new Counter<Tree<MyRule>>();
//			for (Tree<MyRule> tree : allTrees) {
//				for (Tree<MyRule>)
//					double score = treeScore(tree);
//					int count = count(tree, rule);
//					counts.incrementCount() += score * count / Z;
//					norm += score / Z;
//				}
//			}
			return 0.0;
		}
	}

	/**
	 * @param allTrees
	 * @return
	 */
	private List<Tree<StateSet>> genAllNonTerminals(List<Tree<MyRule>> allTrees) {
		List<Tree<StateSet>> retVal = new ArrayList<Tree<StateSet>>();
		for (Tree<MyRule> tree : allTrees)
		{
			for (Tree<MyRule> r : tree.getPreOrderTraversal())
			{
				retVal.add(new Tree<StateSet>(new StateSet(r.getLabel().rule.parentState,(short)1)));
			}
		}
		return retVal;
	}

	/**
	 * @param tree
	 * @param rule
	 * @return
	 */
	private int count(Tree<MyRule> tree, MyRule rule) {
		int count = 0;
		for (Tree<MyRule> node : tree.getPostOrderTraversal()) {
			if (node.getLabel().isSameRule(rule)) {
				count += 1;
			}
		}
		return count;
	}

	/**
	 * @param tree
	 * @return
	 */
	private double treeScore(Tree<MyRule> tree) {
		double score = 1.0;
		for (Tree<MyRule> node : tree.getPreOrderTraversal()) {
			score *= node.getLabel().score;
		}
		return score;
	}

	/**
	 * @param tree
	 * @param goldNode
	 * @return
	 */
	private boolean hasNode(Tree<MyRule> tree, Tree<StateSet> goldNode) {
		StateSet n = goldNode.getLabel();
		for (Tree<MyRule> rNode : tree.getPostOrderTraversal()) {
			if (rNode.getLabel().i == n.from
					&& rNode.getLabel().j == n.to
					&& (rNode.getLabel().tag == n.getState() || (rNode.getLabel().rule != null && rNode
							.getLabel().rule.getParentState() == n.getState()))) {
				return true;
			}
		}
		return false;
	}

	private static class MyRule {
		private double score;

		private Rule rule;

		private short tag = -1;

		private int i;

		private int j;

		private String word;

		public MyRule(double score, Rule rule, int i, int j) {
			this.score = score;
			this.rule = rule;
			this.i = i;
			this.j = j;
		}

		public MyRule(double score, short tag, String word, int i, int j) {
			this.score = score;
			this.tag = tag;
			this.j = j;
			this.i = i;
			this.word = word;
		}

		@Override
		public String toString() {
			if (rule == null) {
				return Numberer.getGlobalNumberer("tags").object(tag) + "->" + word
						+ " " + score + " (" + i + "," + j + ")";
			} else {
				return rule.toString() + " (" + i + "," + j + ")";
			}
		}

		public boolean isSameRule(MyRule r) {
			if (rule != null && r.rule != null) {
				return r.rule.equals(rule);
			} else {
				if (rule == null && r.rule == null) {
					return tag == r.tag && word.equals(r.word);
				}
			}
			return false;

		}

	}

	private List<Tree<MyRule>> getAllTrees(List<String> sentence, short pState,
			int i, int j) {
		return getAllTrees(sentence, pState, i, j, true);
	}

	/**
	 * 
	 * @param sentence
	 * @return
	 */
	private List<Tree<MyRule>> getAllTrees(List<String> sentence, short pState,
			int i, int j, boolean atUnaryPass) {
		final int lastState = numSubStatesArray.length;
		List<Tree<MyRule>> retVal = new ArrayList<Tree<MyRule>>();

		if (j - i == 1 && !grammarTags[pState]) {

			retVal.add(new Tree<MyRule>(new MyRule(lexicon.score(sentence.get(i),
					pState, i, true, false)[0], pState, sentence.get(i), i, j)));

		} else {

			final int nParentSubStates = numSubStatesArray[pState];
			BinaryRule[] parentRules = grammar.splitRulesWithP(pState);
			for (int p = 0; p < parentRules.length; p++) {
				BinaryRule r = parentRules[p];
				for (int k = i + 1; k < j; ++k) {
					final List<Tree<MyRule>> lTrees = getAllTrees(sentence,
							r.leftChildState, i, k, true);
					final List<Tree<MyRule>> rTrees = getAllTrees(sentence,
							r.rightChildState, k, j, true);
					for (Tree<MyRule> lTree : lTrees) {

						for (Tree<MyRule> rTree : rTrees) {
							final ArrayList<Tree<MyRule>> children = new ArrayList<Tree<MyRule>>();

							final Tree<MyRule> newTree = new Tree<MyRule>(new MyRule(
									r.scores[0][0][0], r, i, j));
							newTree.setChildren(children);
							children.add(lTree);
							children.add(rTree);
							retVal.add(newTree);
						}

					}
				}
			}
			if (atUnaryPass)

			{
				UnaryRule[] unaries = grammar.getClosedSumUnaryRulesByParent(pState);
				for (int r = 0; r < unaries.length; r++) {
					UnaryRule ur = unaries[r];
					for (Tree<MyRule> subTree : getAllTrees(sentence, ur.childState, i,
							j, false)) {
						final ArrayList<Tree<MyRule>> children = new ArrayList<Tree<MyRule>>();

						final Tree<MyRule> newTree = new Tree<MyRule>(new MyRule(
								ur.scores[0][0], ur, i, j));
						newTree.setChildren(children);
						children.add(subTree);

						retVal.add(newTree);
					}
				}
			}
		}
		return retVal;

	}

	private void initializePosteriorChart(List<String> sentence,
			boolean noSmoothing, List<String> posTags) {
		final boolean useGoldPOS = (posTags != null);
		int start = 0;
		int end = start + 1;
		oPosteriorScale[0][sentence.size()][0] = -iScale[0][sentence.size()][0];
		assert !ScalingTools.isBadScale(oPosteriorScale[0][sentence.size()][0]);
		oPosteriorScorePreU[0][sentence.size()][0][0] = 1.0 / iScorePostU[0][sentence
				.size()][0][0];
		// oPosteriorScorePreU[0][sentence.size()][0][0] = 1.0;
		for (String word : sentence) {
			end = start + 1;
			int goldTag = -1;
			if (useGoldPOS)
				goldTag = tagNumberer.number(posTags.get(start));

			for (short tag = 0; tag < numSubStatesArray.length; tag++) {
				if (allowedSubStates[start][end][tag] == null)
					continue;
				if (grammarTags[tag])
					continue;
				if (useGoldPOS && tag != goldTag)
					continue;

				// narrowRExtent[start][tag] = end;
				// narrowLExtent[end][tag] = start;
				// wideRExtent[start][tag] = end;
				// wideLExtent[end][tag] = start;
				double[] lexiconScores = lexicon.score(word, tag, start, noSmoothing,
						false);
				// if (!logProbs) iScale[start][end][tag] = scaleArray(lexiconScores,0);
				// iScale[start][end][tag] = 0;
				boolean inTree = inTree(goldTree, start, end, tag, false, true);

				if (inTree) {
					iPosteriorScale[start][end][tag] = -oScale[start][end][tag];
					assert !ScalingTools.isBadScale(iPosteriorScale[start][end][tag]) ;
					double denom = 0.0;
					for (short n = 0; n < lexiconScores.length; n++) {

						// iPosteriorScorePreU[start][end][tag][n] =
						// iScorePreU[start][end][tag][n];
						if (oScorePostU[start][end][tag][n] != 0.0) {
							iPosteriorScorePreU[start][end][tag][n] = iScorePostU[start][end][tag][n];
							denom +=(iScorePostU[start][end][tag][n] * oScorePostU[start][end][tag][n]);
//							 System.out.println("Adding inside for " +
//							 edu.berkeley.nlp.util.Numberer.getGlobalNumberer("tags").object(tag)
//							 + " at lexical phase for span " + start + " " + end);

							// iPosteriorScorePreU[start][end][tag][n] = 1.0 /
							// oScorePostU[start][end][tag][n];
						}
					}
					assert denom > 0.0;
					ArrayUtil.multiplyInPlace(iPosteriorScorePreU[start][end][tag], 1.0 / denom);
				}
				/*
				 * if (start==1){ System.out.println(word+" +TAG
				 * "+(String)tagNumberer.object(tag)+"
				 * "+Arrays.toString(lexiconScores)); }
				 */
			}
			start++;
		}
	}

	private static int matchArrayScales(double[] array1, int scale1,
			double[] array2, int scale2) {
		assert !ScalingTools.isBadScale(scale1) 
				|| !ScalingTools.isBadScale(scale2) ;
		if (scale1 == Integer.MAX_VALUE) scale1 = Integer.MIN_VALUE;
		if (scale2 == Integer.MAX_VALUE) scale2 = Integer.MIN_VALUE;
		
		
		if (scale1 != scale2) {
			int newScale = Math.max(scale1, scale2);
			ScalingTools.scaleArrayToScale(array2, scale2, newScale);
			ScalingTools.scaleArrayToScale(array1, scale1, newScale);
			return newScale;
		}
		return scale1;
	}

	private static int matchArrayScales(double[] array1, int scale1,
			double[] array2, int scale2, double[] array3, int scale3) {
		assert !ScalingTools.isBadScale(scale1) 
				|| !ScalingTools.isBadScale(scale2)
				|| !ScalingTools.isBadScale(scale3) ;
		if (scale1 == Integer.MAX_VALUE) scale1 = Integer.MIN_VALUE;
		if (scale2 == Integer.MAX_VALUE) scale2 = Integer.MIN_VALUE;
		if (scale3 == Integer.MAX_VALUE) scale3 = Integer.MIN_VALUE;
		if (scale1 != scale2 || scale1 != scale3) {
			int newScale = Math.max(Math.max(scale1, scale2), scale3);
			ScalingTools.scaleArrayToScale(array2, scale2, newScale);
			ScalingTools.scaleArrayToScale(array3, scale3, newScale);
			ScalingTools.scaleArrayToScale(array1, scale1, newScale);
			return newScale;
		}
		return scale1;
	}




}
