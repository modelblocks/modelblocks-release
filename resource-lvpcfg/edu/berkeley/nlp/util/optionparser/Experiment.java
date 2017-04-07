package edu.berkeley.nlp.util.optionparser;

import fig.exec.Execution;
import edu.berkeley.nlp.util.Logger;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Dec 8, 2008
 */
public abstract class Experiment {

  public static void run(String[] args,
                         Runnable experiment,
                         boolean useFig,
                         boolean ignoreUnknownFigOpts)
  {
    if (useFig) {
      Execution.ignoreUnknownOpts = ignoreUnknownFigOpts;
      Logger.setFig();
      Execution.init(args,experiment);
    }
    else {
      Logger.startTrack("Starting Experiment:" + experiment.getClass().getSimpleName());
    }
    GlobalOptionParser.registerArgs(args,experiment.getClass());
    GlobalOptionParser.fillOptions(experiment);
    experiment.run();
    if (useFig) Execution.finish();
    else Logger.endTrack();
  }
}
