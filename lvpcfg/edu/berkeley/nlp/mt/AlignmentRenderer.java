package edu.berkeley.nlp.mt;



/**
 * Utilities for rendering a word alignment.
 *
 * @author Dan Klein
 */
public class AlignmentRenderer {
  private static void appendHeader(StringBuilder sb, SentencePair sentencePair) {
    sb.append("\\begin{pspicture}(-1, -7)("+(sentencePair.getEnglishWords().size()+1)+","+(sentencePair.getForeignWords().size()+7)+")\n");
  }

  private static void appendFooter(StringBuilder sb) {
    sb.append("\\end{pspicture}\n");
  }

  private static void appendGrid(StringBuilder sb, SentencePair sentencePair, Alignment goldAlignment, Alignment guessedAlignment) {
    int englishLength = sentencePair.getEnglishWords().size();
    int ForeignLength = sentencePair.getForeignWords().size();
    for (int englishI = 0; englishI < englishLength; englishI++) {
      sb.append("% GRID LINE "+englishI+"\n");
      for (int ForeignI = 0; ForeignI < ForeignLength; ForeignI++) {
        boolean guessed = guessedAlignment.containsSureAlignment(englishI, ForeignI);
        boolean sure = goldAlignment.containsSureAlignment(englishI, ForeignI);
        boolean possible = goldAlignment.containsPossibleAlignment(englishI, ForeignI);
        // render guess
        double radius = 0.0;
        if (guessed) {
          radius = 0.25;
          sb.append("\\psframe[linewidth=0pt,fillstyle=solid,fillcolor=black]("+
                  (englishI+0.5-radius)+", "+
                  (ForeignLength - ForeignI - 0.5 - radius)+")("+
                  (englishI+0.5+radius)+", "+
                  (ForeignLength - ForeignI - 0.5 + radius)+") ");
        }
        // render target
        if (sure) {
          radius = 0.4;
          sb.append("\\psframe[linewidth=1pt,fillstyle=none]("+
                  (englishI+0.5-radius)+", "+
                  (ForeignLength - ForeignI - 0.5 - radius)+")("+
                  (englishI+0.5+radius)+", "+
                  (ForeignLength - ForeignI - 0.5 + radius)+") ");
        } else if (possible) {
          radius = 0.4;
          sb.append("\\psframe[linewidth=1pt,fillstyle=none,framearc=0.5]("+
                  (englishI+0.5-radius)+", "+
                  (ForeignLength - ForeignI - 0.5 - radius)+")("+
                  (englishI+0.5+radius)+", "+
                  (ForeignLength - ForeignI - 0.5 + radius)+") ");
        } else {
          radius = 0.001;
          sb.append("\\psframe[linewidth=0.5pt,fillstyle=none]("+
                  (englishI+0.5-radius)+", "+
                  (ForeignLength - ForeignI - 0.5 - radius)+")("+
                  (englishI+0.5+radius)+", "+
                  (ForeignLength - ForeignI - 0.5 + radius)+") ");
        }
      }
      sb.append("\n");
    }
  }

  private static void appendWords(StringBuilder sb, SentencePair sentencePair) {
    int englishLength = sentencePair.getEnglishWords().size();
    int ForeignLength = sentencePair.getForeignWords().size();
    for (int englishI = 0; englishI < sentencePair.getEnglishWords().size(); englishI++) {
      String englishWord = sentencePair.getEnglishWords().get(englishI);
      sb.append("\\rput[Br]{90}("+(englishI+0.66)+","+(-1)+"){\\huge \\texttt{"+englishWord+"}}\n");
    }
    for (int ForeignI = 0; ForeignI < sentencePair.getForeignWords().size(); ForeignI++) {
      String ForeignWord = sentencePair.getForeignWords().get(ForeignI);
      sb.append("\\rput[Bl]("+(englishLength+0.75)+","+(ForeignLength - ForeignI - 0.66)+"){\\huge \\texttt{"+ForeignWord+"}}\n");
    }
  }

  public static String renderPSTricks(Alignment reference, Alignment proposed, SentencePair sentencePair) {
    StringBuilder sb = new StringBuilder();
    appendHeader(sb, sentencePair);
    appendGrid(sb, sentencePair, reference, proposed);
    appendWords(sb, sentencePair);
    appendFooter(sb);
    return sb.toString();
  }
}
