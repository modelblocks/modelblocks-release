package fig.exec;

import java.io.*;
import java.util.*;
import java.lang.Thread;
import fig.basic.*;
import static fig.basic.LogInfo.*;

/**
 * A separate thread that's responsible for outputting the status
 * of this execution and reading in commands.
 * The thread is actually contained inside.
 */
class MonitorThread implements Runnable {
  private static final int timeInterval = 300; // Number of milliseconds between monitoring
  private boolean stop;
  private Thread thread;

  public MonitorThread() {
    this.stop = false;
    this.thread = new Thread(this);
  }

  void processCommand(String cmd) {
    cmd = cmd.trim();
    if(cmd.equals("")) {
      // Print status
      Execution.getInfo().print(stderr);
      Execution.printOutputMapToStderr();
      StopWatchSet.getStats().print(stderr);
      stderr.println(Execution.getVirtualExecDir());
    }
    else if(cmd.equals("kill")) {
      stderr.println("MonitorThread: KILLING");
      Execution.setExecStatus("killed", true);
      Execution.printOutputMap(Execution.getFile("output.map"));
      throw new RuntimeException("Killed by input command");
    }
    else if(cmd.equals("bail")) {
      // Up to program to look at this flag and actually gracefully stop
      stderr.println("MonitorThread: BAILING OUT");
      Execution.shouldBail = true;
    }
    else
      stderr.println("Invalid command: '" + cmd + "'");
  }

  void readAndProcessCommand() {
    try {
      int nBytes = System.in.available();
      if(nBytes > 0) {
        byte[] bytes = new byte[nBytes];
        System.in.read(bytes);
        String line = new String(bytes);
        processCommand(line);
      }
    } catch(IOException e) {
      // Ignore
    }
  }

  public void run() {
    try {
      while(!stop) {
        if(LogInfo.writeToStdout)
          readAndProcessCommand();

        // Input commands
        Execution.inputMap.readEasy(Execution.getFile("input.map"));

        boolean killed = Execution.create && new File(Execution.getFile("kill")).exists();
        if(killed) Execution.setExecStatus("killed", true);

        // Output status
        Execution.putOutput("log.note", LogInfo.note);
        Execution.putOutput("exec.memory", SysInfoUtils.getUsedMemoryStr());
        Execution.putOutput("exec.time", LogInfo.getWatch().stop().toString());
        Execution.putOutput("exec.errors", "" + LogInfo.getNumErrors());
        Execution.putOutput("exec.warnings", "" + LogInfo.getNumWarnings());
        Execution.setExecStatus("running", false);
        Execution.printOutputMap(Execution.getFile("output.map"));

        if(killed)
          throw new RuntimeException("Killed by 'kill' file");

        Utils.sleep(timeInterval);
      }
    } catch(Exception e) {
      e.printStackTrace();
      System.exit(1); // Die completely
    }
  }

  public void start() {
    thread.start();
  }

  public void finish() {
    stop = true;
    thread.interrupt();
  }
}
