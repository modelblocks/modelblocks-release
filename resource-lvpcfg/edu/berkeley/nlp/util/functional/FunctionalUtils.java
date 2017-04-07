package edu.berkeley.nlp.util.functional;

import edu.berkeley.nlp.util.CollectionUtils;
import fig.basic.Pair;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.*;

/**
 * Collection of Functional Utilities you'd
 * find in any functional programming language.
 * Things like map, filter, reduce, etc..
 *
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Oct 7, 2008
 * Time: 1:06:08 PM
 */
public class FunctionalUtils {


	private static Method getMethod(Class c, String field) {
		Method[] methods = c.getDeclaredMethods();
		String trgMethName = "get" + field;
		Method trgMeth = null;
		for (Method m: methods) {
			if (m.getName().equalsIgnoreCase(trgMethName)) {
				return m;
			}
		}
		return null;				                  
	}

	private static Field getField(Class c, String fieldName) {
		Field[] fields = c.getDeclaredFields();
		for (Field f: fields) {
			if (f.getName().equalsIgnoreCase(fieldName)) {
				return f;
			}
		}
		return null;
	}

	public static<K,I,V> Map<K,V> map(Map<K,I> map, Function<I,V> fn) {
		return map(map,fn, (Predicate<K>) Predicates.getTruePredicate(),new HashMap<K,V>());
	}

	public static<K,I,V> Map<K,V> map(Map<K,I> map, Function<I,V> fn, Predicate<K> pred) {
		return map(map,fn,pred,new HashMap<K,V>());		
	}

	public static<K,I,V> Map<K,V> map(Map<K,I> map, Function<I,V> fn, Predicate<K> pred, Map<K,V> resultMap) {
		for (Map.Entry<K,I> entry: map.entrySet()) {
		  K key = entry.getKey();
		  I inter = entry.getValue();
		  if (pred.apply(key)) resultMap.put(key, fn.apply(inter));
		}
		return resultMap;
	}

  public static<I,O> Map<I,O> mapPairs(Iterable<I> lst, Function<I,O> fn)
  {
    return mapPairs(lst,fn,new HashMap<I,O>());
  }

  public static<I,O> Map<I,O> mapPairs(Iterable<I> lst, Function<I,O> fn, Map<I,O> resultMap)
  {
    for (I input: lst) {
		  O output = fn.apply(input);
		  resultMap.put(input,output);
		}
		return resultMap;
  }

	public static<I,O> List<O> map(Iterable<I> lst, Function<I,O> fn) {
		return map(lst,fn,(Predicate<I>) Predicates.getTruePredicate());
	}

	public static<I,O> List<O> flatMap(Iterable<I> lst,
	                                   Function<I,List<O>> fn) {
		Predicate<I> p = Predicates.getTruePredicate();
		return flatMap(lst,fn,p);
	}


	public static<I,O> List<O> flatMap(Iterable<I> lst,
	                                   Function<I,List<O>> fn,
	                                   Predicate<I> pred) {
		List<List<O>> lstOfLsts = map(lst,fn,pred);
		List<O> init = new ArrayList<O>();
		return reduce(lstOfLsts, init,
				new Function<Pair<List<O>, List<O>>, List<O>>() {
					public List<O> apply(Pair<List<O>, List<O>> input) {
						List<O> result = input.getFirst();
						result.addAll(input.getSecond());
						return result;
					}
				});
	}

	public static<I,O> O reduce(Iterable<I> inputs,	                            
	                            O initial,
	                            Function<Pair<O,I>,O> fn) {
			O output = initial;
			for (I input: inputs) {
				output = fn.apply(Pair.newPair(output,input));
			}
			return output;
	}
	
	public static<I,O> List<O> map(Iterable<I> lst, Function<I,O> fn, Predicate<I> pred) {
			List<O> output = new ArrayList();
			for (I input: lst) {
				if (pred.apply(input)) {
					output.add(fn.apply(input));
				}
			}
			return output;
	}

	public static<I> List<I> filter(Iterable<I> lst, Predicate<I> pred) {
		List<I> output = new ArrayList<I>();
		for (I input: lst) {
		  if (pred.apply(input)) output.add(input);
		}
		return output;
	}

	/**
	 * Groups <code>objs</code> by the field <code>field</code>. Tries
	 * to find public method getField, ignoring case, then to directly
	 * access the field if that fails.
	 * @param objs
	 * @param field
	 * @return
	 */
	public static<K,O> Map<K,List<O>> groupBy(Iterable<O> objs, String field) throws Exception {
		Iterator<O> it = objs.iterator();
		if (!it.hasNext()) return new HashMap<K,List<O>>();
		Class c = it.next().getClass();
		Method trgMeth = getMethod(c, field);
		Field trgField = getField(c, field);
		if (trgMeth == null && trgField == null) {
			throw new RuntimeException("Couldn't find field or method to access " + field);
		}
		Map<K,List<O>> map = new HashMap<K,List<O>>();
		for (O obj: objs) {
			K key = null;
			try {
				key = (K) (trgMeth != null ? trgMeth.invoke(obj) : (K) trgField.get(obj));				
			} catch (Exception e) { throw new RuntimeException(); }
			CollectionUtils.addToValueList(map,key,obj);
		}
		return map;				
	}

  public static <T> T first(Iterable<T> objs, Predicate<T> pred) {
    for (T obj : objs) {
      if (pred.apply(obj)) return obj;
    }
    return null;
  }

	public static<K,I,V> Map<K,V> mapCompose(Map<K,I> map, Function<I,V> meth) {
		Map<K,V> composedMap = new HashMap<K,V>();
		for (Map.Entry<K,I> entry: map.entrySet()) {
		  K key = entry.getKey();
		  I inter = entry.getValue();
			composedMap.put(key,meth.apply(inter));
		}
		return composedMap;
	}

	public static<O,K> List<O> filter(Iterable<O> coll, final String field, final K value) throws Exception {			
		Iterator<O> it = coll.iterator();
		if (!it.hasNext()) return new ArrayList<O>();
		Class c = it.next().getClass();
		final Method m = getMethod(c,field);
		final Field f = getField(c,field);
		return filter(coll, new Predicate<O>()  {
			public Boolean apply(O input) {
				try {
					K inputVal = (K)(m != null ? m.invoke(input) : f.get(input));
					return inputVal.equals(value);
				} catch (Exception e) {  }
				return false;
			}
		});
	}

	/**
	 *   Testing Purposes
	 */
	private static class Person {
		public String prefix ;
		public String name;
		public Person(String name) {
			this.name = name;
			this.prefix = name.substring(0,3);
		}
		public String toString() { return "Person(" + name + ")"; }
	}
		
	public static void main(String[] args) throws Exception {
		List<Person> objs = CollectionUtils.makeList(
			new Person("david"),
			new Person("davs"),
			new Person("maria"),
			new Person("marshia")
		);
		Map<String, List<Person>> grouped = groupBy(objs,"prefix");
		System.out.printf("groupd: %s",grouped);
	}
}
