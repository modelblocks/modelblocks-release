package edu.berkeley.nlp.bp;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.util.functional.Function;
import edu.berkeley.nlp.util.functional.FunctionalUtils;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.math.DoubleArrays;

import java.util.*;

import fig.basic.Pair;

/**
 * User: aria42
 * Date: Jan 25, 2009
 */
public class TreeFactorGraph {

  /**
   * Take a tree and functions to convert into a BP problem
   * and return a tree where each node has a distribution over
   * the labels of itself.
   * @param tree
   * @param varFn
   * @param nodePotentialFn Map Node to Variable
                            Probably put Tree<L> in the id field of Variable
   * @param edgePotentialFn Map Variable to Node Potential
      Can re-use the double[]
      Return null for no potential      
   * @param <L>
   * @return
   */
  public static <L> Pair<List<NodeMarginal>,List<EdgeMarginal>> runBP(
      Tree<L> tree,
      Function<Tree<L>,Variable> varFn,
      Function<Variable, double[]> nodePotentialFn,
      Function<Pair<Variable,Variable>, double[][]> edgePotentialFn)
  {
    // Need Identity Map
    Map<Tree<L>,Variable> varMap = FunctionalUtils.mapPairs(tree,varFn, new IdentityHashMap());
    FactorGraph fg = new FactorGraph(varMap.values());
    for (Tree<L> parent: tree) {
      Variable pvar = varMap.get(parent);
      double[] nodePotentials = nodePotentialFn.apply(pvar);
      FactorPotential nodeFactorPotential = new NodeFactorPotential(nodePotentials);
      fg.addFactor(Collections.singletonList(pvar),nodeFactorPotential);
      for (Tree<L> child : parent.getChildren()) {
        Variable cvar = varMap.get(child);
        double[][] edgePotentials = edgePotentialFn.apply(Pair.newPair(pvar,cvar));
        if (edgePotentials != null) {
          FactorPotential edgeFactorPotential = new EdgeFactorPotential(edgePotentials);
          fg.addFactor(CollectionUtils.makeList(pvar,cvar), edgeFactorPotential);
        }
      }
    }
    BeliefPropogation bp = new BeliefPropogation();
    bp.run(fg);    
    List<NodeMarginal> nodeMarginals = new ArrayList<NodeMarginal>();
    List<EdgeMarginal> edgeMarginals = new ArrayList<EdgeMarginal>();
    for (Factor factor : fg.factors) {
      List<Variable> vars = factor.vars;
      if (factor.potential instanceof NodeFactorPotential) {
        Variable var = vars.get(0);
        double[] marginals = (double[]) factor.marginals;
        nodeMarginals.add(new NodeMarginal(var,marginals));
      }
      else if (factor.potential instanceof EdgeFactorPotential) {
        Variable x = vars.get(0);
        Variable y = vars.get(1);
        double[][] marginals = (double[][]) factor.marginals;
        edgeMarginals.add(new EdgeMarginal(x,y,marginals));
      } else {
        throw new RuntimeException("Unrecognied Factor Potential");
      }
    }
    return Pair.newPair(nodeMarginals,edgeMarginals);
  }
}
  