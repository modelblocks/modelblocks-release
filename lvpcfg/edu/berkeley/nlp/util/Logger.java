package edu.berkeley.nlp.util;

import java.util.Stack;

import fig.basic.LogInfo;

public class Logger
{

	public static interface LogInterface
	{
		public void logs(String s, Object... args);

		public void logss(String s);

		public void startTrack(String s);

		public void endTrack();

		public void dbg(String s);

		public void err(String s);

		public void err(String s, Object... args);

		public void warn(String s);

		public void warn(String string, Object... args);

		public void logss(String string, Object... args);

	}

	public static class FigLogger implements LogInterface
	{

		public void dbg(String s)
		{
			LogInfo.dbg(s);
		}

		public void endTrack()
		{
			LogInfo.end_track();
		}

		public void err(String s)
		{
			LogInfo.error(s);
		}

		public void err(String s, Object... args)
		{
			LogInfo.errors(s, args);
		}

		public void logs(String s, Object... args)
		{
			LogInfo.logs(s, args);
		}

		public void logss(String s)
		{
			LogInfo.logss(s);
		}

		public void logss(String string, Object... args)
		{
			LogInfo.logss(string, args);
		}

		public void startTrack(String s)
		{
			LogInfo.track(s);
		}

		public void warn(String s)
		{
			LogInfo.warning(s);
		}

		public void warn(String string, Object... args)
		{
			LogInfo.warnings(string, args);
		}
	}

	public static class SystemLogger implements LogInterface
	{
		private int trackLevel = 0;

		private Stack<Long> trackStartTimes = new Stack<Long>();

		private String getIndentPrefix()
		{
			StringBuilder builder = new StringBuilder();
			for (int i = 0; i < trackLevel; ++i)
			{
				builder.append("\t");
			}
			return builder.toString();
		}

		private void output(String txt)
		{
			String[] lines = txt.split("\n");
			String prefix = getIndentPrefix();
			for (String line : lines)
			{
				System.out.println(prefix + line);
			}
		}

		public void dbg(String s)
		{
			output(s);
		}

		private String timeString(double milliSecs)
		{
			String timeStr = "";
			int hours = (int) (milliSecs / (1000 * 60 * 60));
			if (hours > 0)
			{
				milliSecs -= hours * 1000 * 60 * 60;
				timeStr += hours + "h";
			}
			int mins = (int) (milliSecs / (1000 * 60));
			if (mins > 0)
			{
				milliSecs -= mins * 1000.0 * 60.0;
				timeStr += mins + "m";
			}
			int secs = (int) (milliSecs / 1000.0);
			//if (secs > 0) {
			//milliSecs -= secs * 1000.0;
			timeStr += secs + "s";
			//}

			return timeStr;
		}

		public void endTrack()
		{
			String timeStr = null;
			synchronized (this)
			{
				trackLevel--;
				double milliSecs = System.currentTimeMillis() - trackStartTimes.peek();
				timeStr = timeString(milliSecs);
			}
			output("} " + (timeStr != null ? "[" + timeStr + "]" : ""));
		}

		public void err(String s)
		{
			System.err.println(s);
		}

		public void logs(String s)
		{
			output(s);
		}

		public void logss(String s)
		{
			output(s);
		}

		public void startTrack(String s)
		{
			output(s + " {");
			synchronized (this)
			{
				trackLevel++;
				trackStartTimes.push(System.currentTimeMillis());
			}
		}

		public void warn(String s)
		{
			System.err.println(s);
		}

		public void logs(String s, Object... args)
		{
			logs(String.format(s, args));
		}

		public void err(String s, Object... args)
		{
			err(String.format(s, args));
		}

		public void warn(String string, Object... args)
		{
			warn(String.format(string, args));
		}

		public void logss(String string, Object... args)
		{
			logss(String.format(string, args));
		}
	}

	private static LogInterface instance = new SystemLogger();

	public static LogInterface i()
	{
		return instance;
	}

	public static void setFig()
	{
		setLogger(new FigLogger());
	}

	public static void setLogger(LogInterface i)
	{
		instance = i;
	}

	// Static Logger Methods
	public static void logs(String s, Object... args)
	{
		i().logs(s, args);
	}

	public static void logss(String s)
	{
		i().logss(s);
	}


	public static void startTrack(String s,Object...args)
	{
		i().startTrack(String.format(s,args));
	}

	public static void endTrack()
	{
		i().endTrack();
	}

	public static void dbg(String s)
	{
		i().dbg(s);
	}

	public static void err(String s)
	{
		i().err(s);
	}

	public static void err(String s, Object... args)
	{
		i().err(s, args);
	}

	public static void warn(String s)
	{
		i().warn(s);
	}

	public static void warn(String string, Object... args)
	{
		i().warn(string, args);
	}

	public void logss(String string, Object... args)
	{
		i().logss(string, args);
	}

}
