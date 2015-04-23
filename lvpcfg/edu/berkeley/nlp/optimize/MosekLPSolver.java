package edu.berkeley.nlp.optimize;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import fig.basic.Pair;

public class MosekLPSolver {

	Counter<Pair> quadraticObjective;
	Counter<Pair> linearObjective ;
	Counter lowerBounds ;
	Counter upperBounds ;
	
	private static enum ConstraintType {LESS_THAN, GREATER_THAN, EQUALITY};
	
	private static class ConstraintInfo {
		ConstraintType type;
		double rhs;
		public ConstraintInfo(ConstraintType type, double rhs) {
			super();
			this.type = type;
			this.rhs = rhs;
		}
		@Override
		public int hashCode() {
			final int PRIME = 31;
			int result = 1;
			long temp;
			temp = Double.doubleToLongBits(rhs);
			result = PRIME * result + (int) (temp ^ (temp >>> 32));
			result = PRIME * result + ((type == null) ? 0 : type.hashCode());
			return result;
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			final ConstraintInfo other = (ConstraintInfo) obj;
			if (Double.doubleToLongBits(rhs) != Double.doubleToLongBits(other.rhs))
				return false;
			if (type == null) {
				if (other.type != null)
					return false;
			} else if (!type.equals(other.type))
				return false;
			return true;
		}
		
	}
	
	Map<Object, List<Pair<Integer,Double>>>  varConstraintMap ; 
	List<ConstraintInfo> constraints = new ArrayList<ConstraintInfo>(); 
	
	public void addLessThanConstraint(Object[] vars, double[] coefs, double rhs) {
		int cIndex = constraints.size();
		ConstraintInfo info = new ConstraintInfo(ConstraintType.LESS_THAN, rhs);
		constraints.add(info);		
		
		for (int i=0; i < coefs.length; ++i) {
			Object var = vars[i];
			double coef = coefs[i];
			CollectionUtils.addToValueList(varConstraintMap, var, new Pair<Integer, Double>(cIndex, coef));
			// TODO This could easily be wrong, since i just changed it randomly
		}
		
		
	}
}
