package edu.berkeley.nlp.optimize;



import edu.berkeley.nlp.util.Counter;

/**
 *
 *  P(C | A, B) = (1-lambda) P(C|A,B) + lambda P(C|A)
 *
 */
public class BackoffDistribution {

	private final Counter<Event>  counter = new Counter<Event>();
	private double lambda = 0.1;

	private final static Object WILD = "*WILD*";

	private static Event queryEvent = null;
	
	private static class Event {
		private Object a = WILD,b = WILD, c = WILD;
		public String toString() {
			return String.format("(%s,%s,%s)",c,a,b);
		}
		private static Event getEvent(Object c, Object a, Object b) {
			if (queryEvent == null) {
				queryEvent = new Event();
			}

			queryEvent.c = c;
			queryEvent.a = a;
			queryEvent.b = b;

			return queryEvent;
		}
		@Override
		public int hashCode() {
			final int PRIME = 31;
			int result = 1;
			result = PRIME * result + ((a == null) ? 0 : a.hashCode());
			result = PRIME * result + ((b == null) ? 0 : b.hashCode());
			result = PRIME * result + ((c == null) ? 0 : c.hashCode());
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
			final Event other = (Event) obj;
			if (a == null) {
				if (other.a != null)
					return false;
			} else if (!a.equals(other.a))
				return false;
			if (b == null) {
				if (other.b != null)
					return false;
			} else if (!b.equals(other.b))
				return false;
			if (c == null) {
				if (other.c != null)
					return false;
			} else if (!c.equals(other.c))
				return false;
			return true;
		}
		
	}

	
	
	
	public void tally(Object c, Object a, Object b) {
		tallyInternal(c, a, b);
		tallyInternal( WILD,a,b );
		tallyInternal( c,a,WILD );
		tallyInternal( WILD,a,WILD );
	}
	
	private double getCount(Object c, Object a, Object b) {		
		return counter.getCount(Event.getEvent(c, a, b));
	}
	
	private void tallyInternal(Object c, Object a, Object b) {
		queryEvent = Event.getEvent(c, a, b);
		if (!counter.containsKey(queryEvent)) {
			queryEvent = new Event();
			Event.getEvent(c,a,b);
			counter.incrementCount(queryEvent, 1.0);
			queryEvent = null;
		} else {
			counter.incrementCount(queryEvent, 1.0);
		}		
	}

	public double getProbability(Object c, Object a, Object b) {
		double nCA = getCount(c,a,WILD);
		if (nCA == 0.0) {
			return 0.0;
		}
		double nA = getCount(WILD,a,WILD) ;
		double nAB = getCount(WILD,a,b) ;
		double nCAB = (nAB > 0 ? getCount(c,a,b)  : 0.0 );

		double pC_AB = (nCAB > 0.0 ? nCAB/nAB : 0.0);
		double pC_A = nCA/nA;
		return (1-lambda) * pC_AB + (lambda) * pC_A;
	}
	
	public static void main(String[] args) {
		BackoffDistribution dist = new BackoffDistribution();
	}
}












