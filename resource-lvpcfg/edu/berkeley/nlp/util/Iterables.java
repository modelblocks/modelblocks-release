package edu.berkeley.nlp.util;

import fig.basic.Pair;

import java.util.Collection;
import java.util.Iterator;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Nov 15, 2008
 */
public class Iterables {
  public static <S,T> Iterable<Pair<S,T>> zip(final Iterable<S> s, final Iterable<T> t) {
    return new Iterable<Pair<S,T>>() {
      public Iterator<Pair<S, T>> iterator() {
        return Iterators.zip(s.iterator(),t.iterator());
      }
    };
  }

  public static <S> Iterable<S> concat(final Iterable<S>...iterables) {
    return new ConcatenationIterable<S>(iterables);
  }

  public static <T> int size(Iterable<T> iterable) {
    if (Collection.class.isInstance(iterable)) {
      return ((Collection) iterable).size();
    }
    int count = 0;
    for (T t : iterable) {
      count++;
    }
    return count;
  }
}
