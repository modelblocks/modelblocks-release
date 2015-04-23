package fig.basic;

import java.util.*;

public class StopWatchSet {
  // For measuring time of certain types of events.
  private static Map<String, StopWatch> stopWatches = new LinkedHashMap<String, StopWatch>();
  private static LinkedList<StopWatch> lastStopWatches = new LinkedList(); // A stack

  public static StopWatch getWatch(String s) {
    return MapUtils.getMut(stopWatches, s, new StopWatch());
  }

  public synchronized static void begin(String s) {
    lastStopWatches.addLast(getWatch(s).start());
  }
  public synchronized static void end() {
    lastStopWatches.removeLast().accumStop();
  }

  public static OrderedStringMap getStats() {
    OrderedStringMap map = new OrderedStringMap();
    for(String key : stopWatches.keySet()) {
      StopWatch watch = getWatch(key);
      map.put(key, watch + " (" + new StopWatch(watch.n == 0 ? 0 : watch.ms/watch.n) + " x " + watch.n + ")");
    }
    return map;
  }
}
