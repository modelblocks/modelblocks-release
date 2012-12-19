package edu.berkeley.nlp.discPCFG;
///**
// * 
// */
//package edu.berkeley.nlp.classify;
//
//import java.io.FileInputStream;
//import java.io.IOException;
//import java.io.ObjectInputStream;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.zip.GZIPInputStream;
//
//import edu.berkeley.nlp.PCFGLA.ConstrainedTwoChartsParser;
//import edu.berkeley.nlp.PCFGLA.Grammar;
//import edu.berkeley.nlp.PCFGLA.Lexicon;
//import edu.berkeley.nlp.PCFGLA.PosteriorConstrainedTwoChartsParser;
//import edu.berkeley.nlp.PCFGLA.SpanPredictor;
//import edu.berkeley.nlp.PCFGLA.StateSetTreeList;
//import edu.berkeley.nlp.syntax.StateSet;
//import edu.berkeley.nlp.syntax.Tree;
//import edu.berkeley.nlp.math.ArrayMath;
//import edu.berkeley.nlp.math.SloppyMath;
//
///**
// * @author adpauls
// * 
// */
//public class PosteriorTrainingObjectiveFunction extends
//		ParsingObjectiveFunction {
//
//	private double allPosteriorsWeight;
//
//	/**
//	 * 
//	 */
//	public PosteriorTrainingObjectiveFunction() {
//		super();
//		// TODO Auto-generated constructor stub
//	}
//
//	/**
//	 * @param gr
//	 * @param lex
//	 * @param trainTrees
//	 * @param sigma
//	 * @param regularization
//	 * @param boost
//	 * @param consName
//	 * @param proc
//	 * @param outName
//	 * @param doGEM
//	 * @param doNotProjectConstraints
//	 */
//	public PosteriorTrainingObjectiveFunction(Linearizer linearizer,
//			StateSetTreeList trainTrees, double sigma, int regularization,
//			double boost, String consName, int proc, String outName, boolean doGEM,
//			boolean doNotProjectConstraints, double allPosteriorsWeight) {
//		super(linearizer, trainTrees, sigma, regularization, boost, consName, proc,
//				outName, doGEM, doNotProjectConstraints,false);
//		this.allPosteriorsWeight = allPosteriorsWeight;
//		// TODO Auto-generated constructor stub
//	}
//
//	@Override
//	protected ParsingObjectiveFunction.Calculator newCalculator(double boost,
//			boolean doNotProjectConstraints, int i) {
//		// TODO Auto-generated method stub
//		return new Calculator(trainingTrees[i],
//				consBaseName, i, grammar, lexicon, dimension, boost, doNotProjectConstraints);
//	}
//
////	 @Override
////	 public double valueAt(double[] x) {
////	 double[] xplush = x.clone();
////	 double h = 1e-4;
////	 int pos = xplush.length - 1;
////	 xplush[pos] += h;
////	 double fplush = super.valueAt(xplush);
////	 double[] derivative = super.derivativeAt(x);
////	 final double f = super.valueAt(x);
////	 double finiteDif = (fplush - f) / h;
////	 System.out.println("Finite Dif: " + finiteDif + " derivative: " +
////	 derivative[pos]);
////	 return f;
////	 }
//
//	class Calculator extends ParsingObjectiveFunction.Calculator {
//
//		@Override
//		protected ConstrainedTwoChartsParser newEParser(Grammar gr, Lexicon lex, SpanPredictor sp,
//				double boost) {
//			// TODO Auto-generated method stub
//			return new PosteriorConstrainedTwoChartsParser(gr, lex, boost);
//		}
//
//		Calculator(StateSetTreeList myT,
//				String consN, int i, Grammar gr, Lexicon lex, int nCounts, double boost,
//				boolean notProject) {
//			// this.nGrWeights = nGrWeights;
//			// this.nCounts = nGrWeights + nLexWeights;
//			// this.consName = consN;
//			// this.myTrees = myT;
//			// this.doNotProjectConstraints = notProject;
//			// this.myID = i;
//			// gParser = new ArrayParser(gr, lex);
//			// eParser = new ConstrainedTwoChartsParser(gr, lex, boost);
//			super(myT, consN, i, gr, lex, null, nCounts, boost, notProject);
//		}
//
//		private void loadConstraints() {
//			myConstraints = new boolean[myTrees.size()][][][][];
//			boolean[][][][][] curBlock = null;
//			int block = 0;
//			int i = 0;
//			if (consName == null)
//				return;
//			for (int tree = 0; tree < myTrees.size(); tree++) {
//				if (curBlock == null || i >= curBlock.length) {
//					int blockNumber = ((block * nProcesses) + myID);
//					curBlock = loadData(consName + "-" + blockNumber + ".data");
//					block++;
//					i = 0;
//					System.out.print(".");
//				}
//				if (!doNotProjectConstraints)
//					eParser.projectConstraints(curBlock[i]);
//				myConstraints[tree] = curBlock[i];
//				i++;
//				if (myConstraints[tree].length != myTrees.get(tree).getYield().size()) {
//					System.out.println("My ID: " + myID + ", block: " + block
//							+ ", sentence: " + i);
//					System.out
//							.println("Sentence length and constraints length do not match!");
//					myConstraints[tree] = null;
//				}
//			}
//
//		}
//
//		private boolean hadZeroPosterior = false;
//		private int numPosteriorsCounted;
//
//		/**
//		 * The most important part of the classifier learning process! This method
//		 * determines, for the given weight vector x, what the (negative) log
//		 * conditional likelihood of the data is, as well as the derivatives of that
//		 * likelihood wrt each weight parameter.
//		 */
//		@Override
//		public Counts call() {
//
//			double myObjective = 0;
//			double[] myECounts = new double[nCounts];
//			double[] myGoldCounts = new double[nCounts];
//			unparsableTrees = 0;
//			incorrectLLTrees = 0;
//			if (ArrayMath.max(PosteriorTrainingObjectiveFunction.this.x) > 200) {
////				myCounts = new Counts(myObjective = -1e10, myECounts, myGoldCounts,
////						unparsableTrees, incorrectLLTrees);
//				return myCounts;
//			}
//
//			if (myConstraints == null)
//				loadConstraints();
//
//			// nInvalidTrees = 0;
//			// done = false;
//			// int maxInvalidTrees = 10;
//			// boolean tooManyInvalidTrees = false;
//			int i = -1;
//			int block = 0;
//			// boolean[][][][][] myConstraints = null;
//			for (Tree<StateSet> stateSetTree : myTrees) {
//				// if(nInvalidTrees>maxInvalidTrees) {
//				// tooManyInvalidTrees = true;
//				// break;
//				// }
//				// compute the ll of the gold tree
//				i++;
//				boolean noSmoothing = true, debugOutput = false, hardCounts = false;
//				// gParser.doInsideOutsideScores(stateSetTree, noSmoothing,
//				// debugOutput);
//
//				// double tree_score = stateSetTree.getLabel().getIScore(0);
//				// int tree_scale = stateSetTree.getLabel().getIScale();
//				// double goldLL = Math.log(tree_score) +
//				// (ScalingTools.LOGSCALE*tree_scale);
//				// if (SloppyMath.isVeryDangerous(goldLL)){
//				// myObjective += -100;
//				// continue;
//				// }
//
//				// parse the sentence
//				List<StateSet> yield = stateSetTree.getYield();
//				List<String> sentence = new ArrayList<String>(yield.size());
//				for (StateSet el : yield) {
//					sentence.add(el.getWord());
//				}
//				boolean[][][][] cons = null;
//				if (consName != null) {
//					cons = myConstraints[i];
//					if (cons.length != sentence.size()) {
//						System.out.println("My ID: " + myID + ", block: " + block
//								+ ", sentence: " + i);
//						System.out.println("Sentence length (" + sentence.size()
//								+ ") and constraints length (" + cons.length
//								+ ") do not match!");
//						System.exit(-1);
//					}
//				}
//
//				eParser.doConstrainedInsideOutsideScores(yield, cons, noSmoothing,
//						stateSetTree, null, false);
//
//				// double tree_score = stateSetTree.getLabel().getIScore(0);
//				// int tree_scale = stateSetTree.getLabel().getIScale();
//				 double[][][] posteriors = new double[sentence.size() + 1][sentence
//				 .size() + 1][eParser.getNumSubStatesArray().length];
//				hadZeroPosterior = false;
//				final double sumLogPosteriors = sumLogPosteriors(stateSetTree,
//						(PosteriorConstrainedTwoChartsParser) eParser, false,posteriors);
//				 double sumLogAllPosteriors = 0.0;
//				if (allPosteriorsWeight != 0.0) {
//					numPosteriorsCounted = 0;
//					sumLogAllPosteriors = sumLogAllPosteriors(
//							(PosteriorConstrainedTwoChartsParser) eParser, false,sentence);
//				}
////				System.out.println("+++ " + numPosteriorsCounted);
//				// eParser.checkScores(stateSetTree);
//				// if (SloppyMath.isVeryDangerous(sumLogPosteriors)) {
//				// // System.out.println("Did bad for " + i);
//				// // myObjective += -1;
//				// continue;
//				// }
//				assert !SloppyMath.isVeryDangerous(sumLogPosteriors);
//				assert !SloppyMath.isVeryDangerous(sumLogAllPosteriors);
////System.out.println(sumLogPosteriors + "::" + sumLogAllPosteriors + "::" + (sumLogAllPosteriors -sumLogPosteriors));
//				myObjective += sumLogPosteriors;
//				myObjective -= allPosteriorsWeight * sumLogAllPosteriors;
//				if (hadZeroPosterior) {
////					myCounts = new Counts(myObjective = -1e10, myECounts, myGoldCounts,
////							unparsableTrees, incorrectLLTrees);
//					return myCounts;
//					// continue;
//				}
//				((PosteriorConstrainedTwoChartsParser) eParser)
//						.doConstrainedPosteriorInsideOutsideScores(sentence, cons,
//								noSmoothing, stateSetTree, null, posteriors);
//				// if (!sanityCheckLLs(goldLL, allLL, sentence, stateSetTree)) {
//				// myObjective += -100;
//				// continue;
//				// }
//				if (i % 100 == 0)
//					System.out.print(".");
//				double[] tmpECounts = new double[nCounts];
//				double[] tmpGoldCounts = new double[nCounts];
//				double[] tmpAllPosteriorCounts = new double[nCounts];
//				double[] tmpAllPosteriorCounts2 = new double[nCounts];
//				double[] tmpGoldCounts2 = new double[nCounts];
//				eParser.incrementExpectedCounts(linearizer, tmpECounts, yield);
//
//				// 0 is C -> B B
//				// 1 is E -> B B
//				// 2 is E -> D D
//				// 3 is X -> V D
//				// 4 is Y -> E D
//				// 5 is Y -> C E
//				// 6 is Y -> C D
//				// 7 is Y -> U D
//				// 8 is X -> C D
//				// 9 is V -> C
//				// 10 is R -> Y
//				// 11 is R -> X
//				// 12 is U -> C
//				// 13 is B -> a
//				// 14 is B -> d
//				// 15 is D -> d
//
//				// System.out.println("....");
//				((PosteriorConstrainedTwoChartsParser) eParser)
//						.incrementExpectedPosteriorGoldCounts(linearizer, tmpGoldCounts,
//								yield, hardCounts);
////				 ((PosteriorConstrainedTwoChartsParser)eParser).incrementExpectedPosteriorGoldCountsByEnumeration(linearizer,
////						 tmpGoldCounts2, sentence, hardCounts,true);
//				if (ArrayMath.max(tmpGoldCounts) > 1e9)
//				{
//					System.out.println("Oy!");
//				}
//
//				if (allPosteriorsWeight > 0.0) {
//					((PosteriorConstrainedTwoChartsParser) eParser)
//							.doConstrainedPosteriorInsideOutsideScores(sentence, cons,
//									noSmoothing, null, null,null);
//					((PosteriorConstrainedTwoChartsParser) eParser)
//							.incrementExpectedPosteriorGoldCounts(linearizer,
//									tmpAllPosteriorCounts, yield, hardCounts);
//				}
//			
////				 ((PosteriorConstrainedTwoChartsParser)eParser).incrementExpectedPosteriorGoldCountsByEnumeration(linearizer,
////						 tmpAllPosteriorCounts2, sentence, hardCounts,false);
//				// double[] xxx = ArrayMath.subtract(tmpGoldCounts,tmpGoldCounts2);
//				// for (int bb = 0; bb < xxx.length; ++bb)
//				// {
//				// xxx[bb] = Math.round(100.0 * xxx[bb]) / 100.0;
//				// System.out.print(xxx[bb] + ",");
//				// }
//				// System.out.println();
//				ArrayMath.multiplyInPlace(tmpECounts, 
//						stateSetTree.getNonTerminals().size() - allPosteriorsWeight * numPosteriorsCounted);
//				ArrayMath.addInPlace(myGoldCounts, ArrayMath.subtract(tmpGoldCounts, ArrayMath.multiply(tmpAllPosteriorCounts,allPosteriorsWeight)));
//				ArrayMath.addInPlace(myECounts, tmpECounts);
//
//				// gParser.incrementExpectedGoldPosteriorCounts(myGoldCounts,
//				// stateSetTree, hardCounts, tree_score, tree_scale,
//				// nGrWeights,posteriors);
//
//			}
//
//			// if (tooManyInvalidTrees||i==0){
//			// return failedSearchResult();
//			// }
//			// System.out.print("done.\nThe objective was "+myObjective);
////			myCounts = new Counts(myObjective, myECounts, myGoldCounts,
////					unparsableTrees, incorrectLLTrees);
//			// done = true;
//			System.out.print(" " + myID + " ");
//			return myCounts;
//		}
//
//		/**
//		 * @param parser
//		 * @param b
//		 * @return
//		 */
//		private double sumLogAllPosteriors(
//				PosteriorConstrainedTwoChartsParser parser, boolean b,
//				List<String> sentence) {
//			double sum = 0.0;
//			for (int start = 0; start < sentence.size(); ++start) {
//				for (int end = start + 1; end <= sentence.size(); ++end) {
//					for (short state = 0; state < grammar.numStates; ++state) {
//						final double summedPosterior = parser.getLogSummedPosterior(state,
//								start, end, false, false);
//						if (summedPosterior != Double.NEGATIVE_INFINITY)
//						{
//							numPosteriorsCounted++;
////						System.out.println(summedPosterior);
//							sum +=(summedPosterior);
//						}
//					}
//				}
//			}
//			return sum;
//		}
//
//		/**
//		 * @param stateSetTree
//		 * @param parser
//		 * @param posteriors
//		 * @return
//		 */
//		private double sumLogPosteriors(Tree<StateSet> stateSetTree,
//				PosteriorConstrainedTwoChartsParser parser, boolean isAfterUnary, double[][][] posteriors) {
//			final StateSet label = stateSetTree.getLabel();
//
//			if (stateSetTree.isLeaf()) {
//
//				final double d = 1.0;
//				 posteriors[label.from][label.to][label.getState()] = d;
//				return Math.log(d);
//			}
//			final boolean currentlyAtUnary = stateSetTree.getChildren().size() == 1
//					&& !stateSetTree.isPreTerminal();
//			// final double summedPosterior =
//			// parser.getSummedPosterior(label.getState(),
//			// label.from, label.to, currentlyAtUnary,isAfterUnary);
//			double summedPosterior = parser.getLogSummedPosterior(label.getState(),
//					label.from, label.to, false, false);
////			assert summedPosterior <= 1.01;
//
//			// System.out.println("summedPosterior for " + stateSetTree + " is " +
//			// summedPosterior);
//			 posteriors[label.from][label.to][label.getState()] = summedPosterior;
//			if (
//					//summedPosterior == 0.0 || 
//					SloppyMath.isVeryDangerous(summedPosterior)) {
//				summedPosterior = 1e-30;
//				hadZeroPosterior = true;
//			}
//			double sum =summedPosterior;
//			for (Tree<StateSet> child : stateSetTree.getChildren()) {
//				sum += sumLogPosteriors(child, parser, currentlyAtUnary,posteriors);
//			}
//			assert !SloppyMath.isVeryDangerous(sum);
//			return sum;
//		}
//
//		@Override
//		public boolean[][][][][] loadData(String fileName) {
//			boolean[][][][][] data = null;
//			try {
//				FileInputStream fis = new FileInputStream(fileName); // Load from file
//				GZIPInputStream gzis = new GZIPInputStream(fis); // Compressed
//				ObjectInputStream in = new ObjectInputStream(gzis); // Load objects
//				data = (boolean[][][][][]) in.readObject(); // Read the mix of grammars
//				in.close(); // And close the stream.
//			} catch (IOException e) {
//				System.out.println("IOException\n" + e);
//				return null;
//			} catch (ClassNotFoundException e) {
//				System.out.println("Class not found!");
//				return null;
//			}
//			return data;
//		}
//
//		// public Counts failedSearchResult(){
//		// double myObjective = 0;
//		// double[] myCounts = new double[nCounts];
//		// double[] myGoldCounts = new double[nCounts];
//		// Arrays.fill(myCounts, 1e-50);
//		// myObjective = Double.POSITIVE_INFINITY;
//		// System.out.println("\n\nTOO MANY INVALID TREES. ABORTING...\n");
//		// return new Counts(myObjective,myCounts,myGoldCounts);
//		// }
//
//		/**
//		 * @param goldLL
//		 * @param allLL
//		 * @param stateSetTree
//		 * @return
//		 */
//		private boolean sanityCheckLLs(double goldLL, double allLL,
//				List<String> sentence, Tree<StateSet> stateSetTree) {
//			// System.out.println("gold ll: "+goldLL+" all ll: "+allLL);
//			if (SloppyMath.isVeryDangerous(allLL)
//					|| SloppyMath.isVeryDangerous(goldLL)) {
//				// System.out.println("Tree is unparsable. allLL:"+allLL+"
//				// goldLL:"+goldLL);//)+"\n"+sentence+"\n"+stateSetTree);
//				unparsableTrees++;
//				return false;
//			}
//			// if (SloppyMath.isVeryDangerous(allLL)) {
//			// System.out.println("Couldn't compute a parse.
//			// allLL:"+allLL);//+"\n"+sentence+"\n"+stateSetTree);
//			// // nInvalidTrees++;
//			// return false;
//			// }
//			// if (SloppyMath.isVeryDangerous(goldLL)) {
//			// System.out.println("Couldn't score the gold parse.
//			// goldLL:"+goldLL);//+"\n"+sentence+"\n"+stateSetTree);
//			// // nInvalidTrees++;
//			// return false;
//			// }
//			if (goldLL - allLL > 1.0e-4) {
//				System.out.println("Something is wrong! The gold LL is " + goldLL
//						+ " and the all LL is " + allLL);// +"\n"+sentence+"\n"+stateSetTree);
//				// System.out.println("Scale: " + stateSetTree.getLabel().getIScale());
//				incorrectLLTrees++;
//				return false;
//			}
//			return true;
//		}
//	}
//
//}
