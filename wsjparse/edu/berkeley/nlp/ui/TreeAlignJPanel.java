package edu.berkeley.nlp.ui;

import java.awt.AlphaComposite;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.syntax.Tree;
import fig.basic.Pair;

public class TreeAlignJPanel extends JPanel
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	int VERTICAL_ALIGN = SwingConstants.CENTER;

	int HORIZONTAL_ALIGN = SwingConstants.CENTER;

	boolean raggedTrees = false;

	boolean writingImageFile = false;

	int maxFontSize = 10;

	int minFontSize = 2;

	int preferredX = 400;

	int preferredY = 300;

	double sisterSkip = 2.5;

	double parentSkip = 0.5;

	double belowLineSkip = 0.1;

	double aboveLineSkip = 0.1;

	double betweenTreeSkip = 2.0;

	FontMetrics myFont;

	Alignment alignment;

	public Alignment getAlignment()
	{
		return alignment;
	}

	public Tree<String> getEnglishTree()
	{
		return alignment.getEnglishTree();
	}

	public Tree<String> getForeignTree()
	{
		return alignment.getForeignTree();
	}

	public void setAlignment(Alignment alignment)
	{
		this.alignment = alignment;
		repaint();
	}

	String nodeToString(Tree<String> t)
	{
		if (t == null) { return " "; }
		Object l = t.getLabel();
		if (l == null) { return " "; }
		String str = (String) l;
		if (str == null) { return " "; }
		return str;
	}

	static class WidthResult
	{
		double width = 0.0;

		double nodeTab = 0.0;

		double nodeCenter = 0.0;

		double childTab = 0.0;
	}

	double width(Alignment alignment, FontMetrics fM)
	{
		if (alignment == null) return preferredY;
		return Math.max(widthResult(alignment.getEnglishTree(), fM).width, widthResult(alignment.getForeignTree(), fM).width);
	}

	public int width()
	{
		return (int) width(alignment, myFont);
	}

	public int height()
	{
		return (int) height(alignment, myFont);
	}

	WidthResult wr = new WidthResult();

	WidthResult widthResult(Tree<String> tree, FontMetrics fM)
	{
		if (tree == null)
		{
			wr.width = 0.0;
			wr.nodeTab = 0.0;
			wr.nodeCenter = 0.0;
			wr.childTab = 0.0;
			return wr;
		}
		double local = fM.stringWidth(nodeToString(tree));
		if (tree.isLeaf())
		{
			wr.width = local;
			wr.nodeTab = 0.0;
			wr.nodeCenter = local / 2.0;
			wr.childTab = 0.0;
			return wr;
		}
		double sub = 0.0;
		double nodeCenter = 0.0;
		double childTab = 0.0;
		for (int i = 0; i < tree.getChildren().size(); i++)
		{
			WidthResult subWR = widthResult(tree.getChildren().get(i), fM);
			if (i == 0)
			{
				nodeCenter += (sub + subWR.nodeCenter) / 2.0;
			}
			if (i == tree.getChildren().size() - 1)
			{
				nodeCenter += (sub + subWR.nodeCenter) / 2.0;
			}
			sub += subWR.width;
			if (i < tree.getChildren().size() - 1)
			{
				sub += sisterSkip * fM.stringWidth(" ");
			}
		}
		double localLeft = local / 2.0;
		double subLeft = nodeCenter;
		double totalLeft = Math.max(localLeft, subLeft);
		double localRight = local / 2.0;
		double subRight = sub - nodeCenter;
		double totalRight = Math.max(localRight, subRight);
		wr.width = totalLeft + totalRight;
		wr.childTab = totalLeft - subLeft;
		wr.nodeTab = totalLeft - localLeft;
		wr.nodeCenter = nodeCenter + wr.childTab;
		return wr;
	}

	double height(Alignment alignment, FontMetrics fM)
	{
		if (alignment == null) return preferredX;
		return height(alignment.getEnglishTree(), fM) + height(alignment.getForeignTree(), fM) + fM.getHeight() * betweenTreeSkip;
	}

	double height(Tree<String> tree, FontMetrics fM)
	{
		if (tree == null) { return 0.0; }
		double depth = tree.getDepth();
		//double f = fM.getHeight() ;
		return fM.getHeight() * (depth * (1.0 + parentSkip + aboveLineSkip + belowLineSkip) - parentSkip);
	}

	FontMetrics pickFont(Graphics2D g2, Alignment alignment, Dimension space)
	{
		Font font = g2.getFont();
		String name = font.getName();
		int style = font.getStyle();

		for (int size = maxFontSize; size > minFontSize; size--)
		{
			font = new Font(name, style, size);
			g2.setFont(font);
			FontMetrics fontMetrics = g2.getFontMetrics();
			if (height(alignment, fontMetrics) > space.getHeight())
			{
				continue;
			}
			if (width(alignment, fontMetrics) > space.getWidth())
			{
				continue;
			}
			//System.out.println("Chose: "+size+" for space: "+space.getWidth());
			return fontMetrics;
		}
		font = new Font(name, style, minFontSize);
		g2.setFont(font);
		return g2.getFontMetrics();
	}

	double paintTree(Tree<String> t, Point2D start, int layerDepth, Point2D.Double[] leaves, List<Tree<String>> preTerminals, Graphics2D g2, FontMetrics fM,
		boolean inverted)
	{
		if (t == null) { return 0.0; }
		String nodeStr = nodeToString(t);
		double nodeWidth = fM.stringWidth(nodeStr);
		double nodeHeight = fM.getHeight();
		double nodeAscent = fM.getAscent();
		WidthResult wr = widthResult(t, fM);
		double treeWidth = wr.width;
		double nodeTab = wr.nodeTab;
		double childTab = wr.childTab;
		double nodeCenter = wr.nodeCenter;
		double layerMultiplier = (1.0 + belowLineSkip + aboveLineSkip + parentSkip);
		double layerHeight = nodeHeight * layerMultiplier;
		double childStartX = start.getX() + childTab;
		double childStartY = start.getY() + (inverted ? -layerHeight : layerHeight);
		double lineStartX = start.getX() + nodeCenter;
		double lineStartY = start.getY() + nodeHeight * (inverted ? -aboveLineSkip : (1.0 + belowLineSkip));
		double lineEndY = lineStartY + nodeHeight * (inverted ? -parentSkip : parentSkip);
		if (raggedTrees || layerDepth == t.getDepth())
		{
			// draw root
			g2.drawString(nodeStr, (float) (nodeTab + start.getX()), (float) (start.getY() + nodeAscent));
			if (t.isLeaf())
			{
				for (int leafIndex = 0; leafIndex < preTerminals.size(); leafIndex++)
				{
					if (preTerminals.get(leafIndex).getChildren().get(0) == t)
						leaves[leafIndex++] = new Point2D.Double(childStartX + nodeCenter, start.getY());
				}
				return nodeWidth;
			}
			// recursively draw children
			for (int i = 0; i < t.getChildren().size(); i++)
			{
				Tree<String> child = t.getChildren().get(i);
				double cWidth = paintTree(child, new Point2D.Double(childStartX, childStartY), layerDepth - 1, leaves, preTerminals, g2, fM, inverted);
				// draw connectors
				wr = widthResult(child, fM);
				double lineEndX = childStartX + wr.nodeCenter;
				g2.draw(new Line2D.Double(lineStartX, lineStartY, lineEndX, lineEndY));
				childStartX += cWidth;
				if (i < t.getChildren().size() - 1)
				{
					childStartX += sisterSkip * fM.stringWidth(" ");
				}
			}
		}
		else
		{
			double lineEndX = lineStartX;
			lineStartY = start.getY() + nodeHeight * (inverted ? (1.0 + belowLineSkip) : 0);
			lineEndY = lineStartY + (inverted ? -layerHeight : layerHeight);
			g2.draw(new Line2D.Double(lineStartX, lineStartY, lineEndX, lineEndY));
			paintTree(t, new Point2D.Double(start.getX(), childStartY), layerDepth - 1, leaves, preTerminals, g2, fM, inverted);
		}
		return treeWidth;
	}

	private void paintWordAlignments(Alignment alignment, Point2D.Double[] englishLeaves, Point2D.Double[] foreignLeaves, FontMetrics fM, Graphics2D g2)
	{
		double nodeHeight = fM.getHeight();
		for (Pair<Integer, Integer> arc : alignment.getSureAlignments())
		{
			if (arc.getFirst() < englishLeaves.length && arc.getSecond() < foreignLeaves.length)
			{
				Point2D.Double englishPoint = englishLeaves[arc.getFirst()];
				Point2D.Double foreignPoint = foreignLeaves[arc.getSecond()];
				double lineStartX = englishPoint.getX();
				double lineStartY = englishPoint.getY() + nodeHeight * (1.0 + belowLineSkip);
				double lineEndX = foreignPoint.getX();
				double lineEndY = foreignPoint.getY() - nodeHeight * aboveLineSkip;
				g2.draw(new Line2D.Double(lineStartX, lineStartY, lineEndX, lineEndY));
			}
		}
	}

	@Override
	public void paint(Graphics g)
	{
		g.clearRect(0, 0, getWidth(), getHeight());
		((Graphics2D) g).setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		this.paintComponent(g);
	}

	@Override
	public void paintComponent(Graphics g)
	{
		if (alignment == null || alignment.getEnglishTree() == null || alignment.getForeignTree() == null) { return; }
		Graphics2D g2 = (Graphics2D) g;
		Dimension space = getSize();
		g2.clearRect(0, 0, space.width, space.height);
		//FontMetrics fM = pickFont(g2, tree, space);

		double englishWidth = widthResult(alignment.getEnglishTree(), myFont).width;
		double foreignWidth = widthResult(alignment.getForeignTree(), myFont).width;
		double height = height(alignment, myFont);
		preferredX = (int) Math.max(englishWidth, foreignWidth);
		preferredY = (int) height;
		setSize(new Dimension(preferredX, preferredY));
		setPreferredSize(new Dimension(preferredX, preferredY));
		setMaximumSize(new Dimension(preferredX, preferredY));
		setMinimumSize(new Dimension(preferredX, preferredY));
		//setSize(new Dimension((int)Math.round(width), (int)Math.round(height)));
		g2.setFont(myFont.getFont());

		space = getSize();
		double englishStartX = 0.0;
		double foreignStartX = 0.0;
		double englishStartY = 0.0;
		if (HORIZONTAL_ALIGN == SwingConstants.CENTER)
		{
			englishStartX = (space.getWidth() - englishWidth) / 2.0;
			foreignStartX = (space.getWidth() - foreignWidth) / 2.0;
		}
		if (HORIZONTAL_ALIGN == SwingConstants.RIGHT)
		{
			englishStartX = space.getWidth() - englishWidth;
			foreignStartX = space.getWidth() - foreignWidth;
		}
		if (VERTICAL_ALIGN == SwingConstants.CENTER)
		{
			englishStartY = (space.getHeight() - height) / 2.0;
		}
		if (VERTICAL_ALIGN == SwingConstants.BOTTOM)
		{
			englishStartY = space.getHeight() - height;
		}
		double foreignStartY = englishStartY + height - myFont.getHeight() * (1.0 + belowLineSkip);
		super.paintComponent(g);

		//	    if (writingImageFile) {
		//	    	Rectangle background = new Rectangle(space.width,space.height);
		//	    	g2.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f));
		//	    
		//	    	g2.setPaint(Color.white);
		//	    	g2.fill(background);
		//	    } else {
		g2.setBackground(Color.white);
		g2.clearRect(0, 0, space.width, space.height);
		//	    }
		g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		g2.setPaint(Color.black);

		Point2D.Double[] englishLeaves = new Point2D.Double[alignment.getEnglishTree().getYield().size()];
		Point2D.Double[] foreignLeaves = new Point2D.Double[alignment.getForeignTree().getYield().size()];

		paintTree(alignment.getEnglishTree(), new Point2D.Double(englishStartX, englishStartY), alignment.getEnglishTree().getDepth(), englishLeaves, alignment
			.getEnglishTree().getPreTerminals(), g2, myFont, false);
		paintTree(alignment.getForeignTree(), new Point2D.Double(foreignStartX, foreignStartY), alignment.getForeignTree().getDepth(), foreignLeaves, alignment
			.getForeignTree().getPreTerminals(), g2, myFont, true);
		paintWordAlignments(alignment, englishLeaves, foreignLeaves, myFont, g2);
	}

	public TreeAlignJPanel()
	{
		this(SwingConstants.CENTER, SwingConstants.CENTER);
	}

	public TreeAlignJPanel(int hAlign, int vAlign)
	{
		HORIZONTAL_ALIGN = hAlign;
		VERTICAL_ALIGN = vAlign;
		//setPreferredSize(new Dimension(preferredX, preferredY));
		Font font = getFont();
		font = new Font("SansSerif", font.getStyle(), maxFontSize);
		myFont = getFontMetrics(font);
	}

	public void setMinFontSize(int size)
	{
		minFontSize = size;
	}

	public void setMaxFontSize(int size)
	{
		maxFontSize = size;
	}

	public static boolean writeImageFile(Alignment alignment, final String filename)
	{
		TreeAlignJPanel tajp = new TreeAlignJPanel();
		tajp.setAlignment(alignment);
		tajp.writingImageFile = filename != null;

		JFrame frame = new JFrame();
		frame.getContentPane().add(tajp, BorderLayout.CENTER);
		frame.addWindowListener(new WindowAdapter()
		{
			@Override
			public void windowClosing(WindowEvent e)
			{

			}
		});
		frame.pack();
		if (!tajp.writingImageFile)
		{
			//Image img = frame.createImage(frame.getWidth(),frame.getHeight());
			frame.setVisible(true);
			frame.setVisible(true);
			frame.setSize(tajp.preferredX, tajp.preferredY);
		}

		if (filename == null) return false;
		int t = 1;
		t++;

		BufferedImage bi = new BufferedImage(tajp.width(), tajp.height(), BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bi.createGraphics();
		//int rule = AlphaComposite.SRC_OVER;
		//AlphaComposite ac = AlphaComposite.getInstance(rule, 0f);
		//g2.setComposite(ac);

		g2.clearRect(0, 0, frame.getWidth(), frame.getHeight());
		g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));

		tajp.paintComponent(g2); //paint the graphic to the offscreen image
		//g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		//g2.drawImage(src, null, 0, 0);
		//g2.dispose();
		try
		{
			ImageIO.write(bi, "png", new File(filename)); //save as png format DONE!
			return true;
		}
		catch (IOException e)
		{
			return false;
		}
	}

}
