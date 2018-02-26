import sys, os, re, math, json, argparse, time
from operator import add
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

argparser = argparse.ArgumentParser('''
Reads in gold and test linetrees files and performs a number of evaluations. The test and gold data must span the same character sequence and sentence segmentation, but token segmentations may differ. Evaluations include (1) Overall (character span) accuracy, segmentation accuracy, bracketing accuracy, constituent labeling, part-of-speech tagging (terminal labeling), and discovery of certain maximal projection types (if the gold standard is PTB-style). NOTE: 
''')
argparser.add_argument('gold', nargs=1, help='Gold linetrees file (PTB)')
argparser.add_argument('test', nargs=1, help='Test linetrees file')
argparser.add_argument('-m', '--minlength', dest='minlength', type=int, default=2, help='Exclude gold NPs shorter than minlength number of words.')
argparser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Print detailed information about each sentence.')
args, unknown = argparser.parse_known_args()

# Extension of tree.Tree() to provide useful character-level functionality
# that permits evaluation and comparison of trees even when their word
# tokenizations differ.
class Char_Tree(tree.Tree,object):
  def __init__(self, c='', ch=[], p=None,l=0,r=0,lc=0,rc=0):
    super(Char_Tree, self).__init__(c,ch,p,l,r)
    self.lc = lc
    self.rc = rc

  # obtain tree from string
  def read(self,s,fIndex=0,cIndex=0):
    self.ch = []
    # parse a string delimited by whitespace as a terminal branch (a leaf)
    m = re.search('^ *([^ ()]+) *(.*)',s)
    if m != None:
        (self.c,s) = m.groups()
        self.l = fIndex
        self.lc = cIndex
        self.r = fIndex
        self.rc = self.lc + len(self.c) - 1
        return s, fIndex+1, cIndex + len(self.c)
    # parse nested parens as a non-terminal branch
    m = re.search('^ *\( *([^ ()]*) *(.*)',s)
    if m != None:
        (self.c,s) = m.groups()
        self.l = fIndex
        self.lc = cIndex
        # read children until close paren
        while True:
            m = re.search('^ *\) *(.*)',s)
            if m != None:
                return m.group(1), fIndex, cIndex
            t = Char_Tree()
            s, fIndex, cIndex = t.read(s, fIndex, cIndex)
            self.ch += [t]
            t.p = self
            self.r = t.r
            self.rc = t.rc
    return ''

# Get dicts of spans, categories, and depths from a given tree
#
#  Params:
#    t: a tree representation (Char_Tree() only, will not work with tree.Tree())
#    spans: a dictionary mapping a 2-tuple of ints (<first_index, last_index>)
#           representing a character span to fields 'wlen' (length in words of 
#           span), 'cat' (category of span), and 'wspan' (a 2-tuple of ints 
#           representing the same span in words)
#    cats: a dictionary mapping categories to a list of the char spans that they label
#    depths: a dictionary mapping char spans to left-corner depths to counts
#    depth: the left-corner depth of the current tree
#    right_parent: the current tree's parent was a right child (boolean)
#
#  Return:
#    spans, cats, and depths (updates of params)
#
#  NOTE:
#    Because Python doesn't garbage-collect the dictionary params once the call
#    terminates, clients should explicitly supply empty dictionaries as param
#    when reinitialization is needed.
def process_tree(t, spans={}, cats={}, depths={}, depth=1, right_parent=False):
  if (t.lc,t.rc) not in spans:
    spans[(t.lc,t.rc)] = {'wlen': t.r-t.l+1, 'cat': t.c, 'wspan':(t.l,t.r)}
  if (t.lc,t.rc) not in depths and t.r-t.l+1 >= args.minlength:
    depths[(t.lc,t.rc)] = depth 
  if t.c not in cats:
    cats[t.c] = [(t.lc,t.rc)]
  else:
    cats[t.c] += [(t.lc,t.rc)]
  for i in range(len(t.ch)):
    if right_parent and i == 0 and len(t.ch) > 1:
      newdepth = depth + 1
    else:
      newdepth = depth
    process_tree(t.ch[i], spans, cats, depths, newdepth, i>0)
  return spans, cats, depths
     
  
# Get a sequential list of the preterminal nodes in a tree
#
#  Params:
#    t: a tree representation (tree.Tree() or Char_Tree())
#
#  Return:
#    a list of the preterminal nodes in t
def preterms(t):
  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    return [t.c]
  else:
    tmp = []
    for ch in t.ch:
      tmp += preterms(ch)
    return tmp

# Get an integer representation of a tag sequence.
# Maps each unique tag type in tags to a unique
# integer label in the output.
#
#  Params:
#    tags: a sequential list of tags
#
#  Return:
#    ints: an integer representation of tags
def tags2ints(tags):
  legend = {}
  ints = []
  i = 1
  for a in tags:
    if a in legend:
      ints.append(legend[a])
    else:
      ints.append(i)
      legend[a] = i
      i += 1
  return ints

# Get a binary segmentation list from a word list.
# Maps each non-space character to 1 (first character in
# in a segment) or 0 (otherwise).
#
#  Params:
#    wrds: a list of the words/tokens in a sequence
#
#  Return:
#    A binary segmentation list representation of wrds
def get_seg(wrds):
  out = []
  chars = ' '.join(wrds)
  seg = True
  for i in range(len(chars)):
    if (chars[i]) == ' ':
      seg = True
    else:
      out.append(int(seg))
      seg = False
  return out

# ACCURACY EVAL FUNCTIONS:

# Get precision, recall, and F1
#
#  Params:
#    tp: count of true positives
#    fp: count of false positives
#    fn: count of false negatives
#
#  Return:
#    p: precision
#    r: recall
#    f1: f-measure
def accuracy(tp, fp, fn):
  if tp + fp + fn == 0:
    return 1, 1, 1
  elif tp == 0:
    return 0, 0, 0
  else:
    p = float(tp) / float(tp + fp)
    r = float(tp) / float(tp + fn)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1

# Get marginal entropy of a set of classes
#
#  Params:
#    A: dictionary from classes (C) to clusters (K) to counts
#    n: total number of data points
#
#  Return:
#    H(C) (a real number)
def H_C(A, n):
  H = float(0)
  for c in A:
    exp_c = float(sum([A[c][k] for k in A[c]])) / n
    H += exp_c * math.log(exp_c)
  return -H

# Get conditional entropy of a set of classes given
# a set of clusters
#
#  Params:
#    A: a dictionary from clusters (K) to classes (C) to counts
#    n: total number of data points
#
#  Return:
#    H(C|K) (a real number)
def H_CgivK(A, n):
  H = float(0)
  for k in A:
    denom = float(sum([A[k][c] for c in A[k]]))
    for c in A[k]:
      H += float(A[k][c])/n * math.log(float(A[k][c])/denom)
  return -H

# Get homogeneity score h (Rosenberg & Hirschberg 2007)
#
#  Params:
#    A: a dictionary from clusters (K) to classes (C)
#    Aprime: a dictionary from classes (C) to clusters (K)
#    n: total number of data points
def homog(A, Aprime, n):
  hc = H_C(Aprime, n)
  if hc == 0:
    return 1
  else:
    return 1 - (H_CgivK(A, n) / hc)

# Get many-to-one accuracy (raw accuracy given a mapping
# of clusters to most frequent classes)
#
#  Params:
#    A: a dictionary from classes (C) to clusters (K)
#
#  Return:
#    many-to-one accuracy (a real number)
def m21(A):
  correct = 0
  total = 0
  for c in A:
    max = 0
    argMax = None
    for k in A[c]:
      total += A[c][k]
      if A[c][k] > max:
        max = A[c][k]
        argMax = k
    correct += max
  return float(correct) / total

# Main program
def main():
  # Initialize global variables
  g = Char_Tree()
  t = Char_Tree()
  all_tp = all_fp = all_fn = 0
  brac_tp = brac_fp = brac_fn = 0
  seg_tp = seg_fp = seg_fn = 0
  n_wrds = 0
  max_proj = ['NP', 'VP', 'ADJP', 'ADVP', 'PP', 'S']
  max_proj_scores = {}
  for cat in max_proj:
    max_proj_scores[cat] = {}
    max_proj_scores[cat]['hit'] = 0
    max_proj_scores[cat]['miss'] = 0
  sentid = 1
  same_g2t = {}
  same_t2g = {}
  t_cat_counts = {}
  leaves_g2t = {}
  leaves_t2g = {}
  depth_counts = {}
  depth_length_counts = {}

  # Open connections to gold and test datasets
  with open(args.gold[0], 'r') as gold:
    with open(args.test[0], 'r') as test:
      # Read the next line from each dataset
      g_line = gold.readline()
      while g_line and g_line.strip() == '':
        g_line = gold.readline()

      t_line = test.readline()
      while t_line and t_line.strip() == '':
        t_line = test.readline()

      # Process each pair of trees
      while g_line and t_line:
        # Read in trees and get word and character lists from them
        g.read(g_line)
        g_wrds = g.words()
        g_chrs = ''.join(g_wrds)
        t.read(t_line)
        t_wrds = t.words()
        t_chrs = ''.join(t_wrds)
        if t_line.startswith('(FAIL'):
          g_line = gold.readline()
          t_line = test.readline()
          continue
        assert g_chrs == t_chrs, 'Gold sentence (%s) differs from test sentence (%s)' %(' '.join(g.words()), ' '.join(t.words()))
        
        # Get data about character spans and their labels
        seg_g = get_seg(g_wrds)
        g_cat_by_spans, g_spans_by_cat = process_tree(g, {}, {}, {})[:2]
        t_cat_by_spans, t_spans_by_cat, t_depth_by_spans = process_tree(t, {}, {}, {})
        g_spans = g_cat_by_spans.keys()
        t_spans = t_cat_by_spans.keys()
        all_same = set(g_spans) & set(t_spans)

        # Get list of terminals in test and gold
        g_leaves = [span for span in g_spans if g_cat_by_spans[span]['wlen'] == 1]
        t_leaves = [span for span in t_spans if t_cat_by_spans[span]['wlen'] == 1]
        leaves_same = set(g_leaves) & set(t_leaves)
        n_wrds += len(leaves_same)

        # Get list of complex non-terminals in test and gold
        g_spans_complex = [span for span in g_spans if g_cat_by_spans[span]['wlen'] >= args.minlength]
        t_spans_complex = [span for span in t_spans if t_cat_by_spans[span]['wlen'] >= args.minlength]
        # Remove constituents over bad token segmentations (i.e. all constituents
        # whose edges fall on characters that are not segment boundaries in the gold).
        # Avoids double-punishing the bracketing for segmentation errors.
        t_spans_complex = [span for span in t_spans_complex if seg_g[span[0]] == 1 and (span[1] == len(seg_g) - 1 or seg_g[span[1]+1] == 1)]
        # Remove spans from test that were analyzed as complex but
        # correspond to leaves in the gold.
        # Avoids punishing the bracketing for oversegmentation.
        t_spans_complex = set(t_spans_complex) - set(g_leaves)
        for span in [span for span in t_spans if t_cat_by_spans[span]['wlen'] >= args.minlength]:
          if t_cat_by_spans[span]['cat'] in t_cat_counts:
            t_cat_counts[t_cat_by_spans[span]['cat']] += 1
          else:
            t_cat_counts[t_cat_by_spans[span]['cat']] = 1
        brac_same = set(g_spans_complex) & set(t_spans_complex)
        
        # Get depth counts
        depth_counts_sent = {}
        depth_length_counts_sent = {}
        for span in t_depth_by_spans:
          cur_depth = t_depth_by_spans[span]
          span_length = t_cat_by_spans[span]['wlen']
          if args.debug:
	    if cur_depth in depth_counts_sent:
	      depth_counts_sent[cur_depth] += 1
	    else:
	      depth_counts_sent[cur_depth] = 1
          if cur_depth in depth_counts:
            depth_counts[cur_depth] += 1
          else:
            depth_counts[cur_depth] = 1
          if args.debug:
	    if cur_depth in depth_length_counts_sent:
	      if span_length in depth_length_counts_sent[cur_depth]: 
		depth_length_counts_sent[cur_depth][span_length] += [' '.join(t_wrds[t_cat_by_spans[span]['wspan'][0]:t_cat_by_spans[span]['wspan'][1]+1])]
	      else:
		depth_length_counts_sent[cur_depth][span_length] = [' '.join(t_wrds[t_cat_by_spans[span]['wspan'][0]:t_cat_by_spans[span]['wspan'][1]+1])]
	    else:
	      depth_length_counts_sent[cur_depth] = {span_length: [' '.join(t_wrds[t_cat_by_spans[span]['wspan'][0]:t_cat_by_spans[span]['wspan'][1]+1])]}
          if cur_depth in depth_length_counts:
            if span_length in depth_length_counts[cur_depth]: 
              depth_length_counts[cur_depth][span_length] += 1
            else:
              depth_length_counts[cur_depth][span_length] = 1
          else:
            depth_length_counts[cur_depth] = {span_length: 1}


        # Get maximal projection discovery scores
        if args.debug:
          debug_maxproj = 'Maximal projection discovery:\n'
        for cat in max_proj_scores.keys():
          if cat in g_spans_by_cat:
            for span in g_spans_by_cat[cat]:
              if g_cat_by_spans[span]['wlen'] >= args.minlength:
                if args.debug:
                  debug_maxproj += '  ' + cat + ' span: ' + ' '.join(t_wrds[g_cat_by_spans[span]['wspan'][0]:g_cat_by_spans[span]['wspan'][1] + 1]) + '\n'
                if span in t_cat_by_spans:
                  max_proj_scores[cat]['hit'] += 1
                  if args.debug:
                    debug_maxproj += '    Found: YES\n'
                else:
                  max_proj_scores[cat]['miss'] += 1
                  if args.debug:
                    debug_maxproj += '    Found: NO\n'

        # Update cluster-label matrix (constituents)
        for span in brac_same:
          if g_cat_by_spans[span]['cat'] in same_g2t:
            if t_cat_by_spans[span]['cat'] in same_g2t[g_cat_by_spans[span]['cat']]:
              same_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] += 1
            else:
              same_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] = 1
          else:
            same_g2t[g_cat_by_spans[span]['cat']] = {t_cat_by_spans[span]['cat']: 1}
          if t_cat_by_spans[span]['cat'] in same_t2g:
            if g_cat_by_spans[span]['cat'] in same_t2g[t_cat_by_spans[span]['cat']]:
              same_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] += 1
            else:
              same_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] = 1
          else:
            same_t2g[t_cat_by_spans[span]['cat']] = {g_cat_by_spans[span]['cat']: 1}

        # Update cluster-tag matrix (parts-of-speech)
        for span in leaves_same:
          if g_cat_by_spans[span]['cat'] in leaves_g2t:
            if t_cat_by_spans[span]['cat'] in leaves_g2t[g_cat_by_spans[span]['cat']]:
              leaves_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] += 1
            else:
              leaves_g2t[g_cat_by_spans[span]['cat']][t_cat_by_spans[span]['cat']] = 1
          else:
            leaves_g2t[g_cat_by_spans[span]['cat']] = {t_cat_by_spans[span]['cat']: 1}
          if t_cat_by_spans[span]['cat'] in leaves_t2g:
            if g_cat_by_spans[span]['cat'] in leaves_t2g[t_cat_by_spans[span]['cat']]:
              leaves_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] += 1
            else:
              leaves_t2g[t_cat_by_spans[span]['cat']][g_cat_by_spans[span]['cat']] = 1
          else:
            leaves_t2g[t_cat_by_spans[span]['cat']] = {g_cat_by_spans[span]['cat']: 1}

        # Update scores for overall (constituent span) accuracy
        all_tp += len(all_same)
        all_t_only = len(t_spans) - len(all_same)
        all_fp += all_t_only
        all_g_only = len(g_spans) - len(all_same)
        all_fn += all_g_only

        # Update scores for segmentation
        seg_tp += len(leaves_same)
        seg_t_only = len(t_leaves) - len(leaves_same)
        seg_fp += seg_t_only
        seg_g_only = len(g_leaves) - len(leaves_same)
        seg_fn += seg_g_only

        # Update scores for bracketing accuracy
        brac_tp += len(brac_same)
        brac_t_only = len(t_spans_complex) - len(brac_same)
        brac_fp += brac_t_only
        brac_g_only = len(g_spans_complex) - len(brac_same)
        brac_fn += brac_g_only

        # Print debugging/verbose log for this sentence
        if args.debug:
          all_p, all_r, all_f1 = accuracy(len(all_same), all_t_only, all_g_only)
          seg_p, seg_r, seg_f1 = accuracy(len(leaves_same), seg_t_only, seg_g_only)
          brac_p, brac_r, brac_f1 = accuracy(len(brac_same), brac_t_only, brac_g_only)
          print('=================================')
          print('Evaluating sentence #%d:' %sentid)
          print('Unsegmented input: %s' %g_chrs)
          print('Gold segmentation: %s' %' '.join(g_wrds))
          print('Gold tree: %s' %str(g))
          print('Test segmentation: %s' %' '.join(t_wrds))
          print('Test tree: %s' %str(t))
          print('')
          print('Overall (char spans):')
          print('  Num gold character spans: %d' %len(g_spans))
          print('  Num test character spans: %d' %len(t_spans))
          print('  Matching: %d' %len(all_same))
          print('  False positives (in test but not gold): %d' %all_t_only)
          print('  False negatives (in test but not gold): %d' %all_g_only)
          print('  Precision: %.4f' %all_p)
          print('  Recall: %.4f' %all_r)
          print('  F-Measure: %.4f' %all_f1)
          print('')
          print('Segmentation:')
          print('  Num gold words: %d' %len(g_leaves))
          print('  Num test words: %d' %len(t_leaves))
          print('  Matching: %d' %len(leaves_same))
          print('  False positives (in test but not gold): %d' %seg_t_only)
          print('  False negatives (in test but not gold): %d' %seg_g_only)
          print('  Precision: %.4f' %seg_p)
          print('  Recall: %.4f' %seg_r)
          print('  F-Measure: %.4f' %seg_f1)
          print('')
          print('Bracketing:')
          print('  (Ignoring constituents of length < %d)' %args.minlength)
          print('  Gold constituents:\n    ' + '\n    '.join([' '.join(g_wrds[g_cat_by_spans[x]['wspan'][0]:g_cat_by_spans[x]['wspan'][1]+1]) for x in g_spans_complex]))
          print('  Test constituents:\n    ' + '\n    '.join([' '.join(t_wrds[t_cat_by_spans[x]['wspan'][0]:t_cat_by_spans[x]['wspan'][1]+1]) for x in t_spans_complex]))
          print('  Test constituents by depth:')
          for depth in depth_counts_sent:
            print('    Depth = %d:\n      ' %depth + '\n      '.join([' '.join(t_wrds[t_cat_by_spans[x]['wspan'][0]:t_cat_by_spans[x]['wspan'][1]+1]) for x in t_depth_by_spans if t_depth_by_spans[x] == depth]))
            print('      Depth = %d counts by span length (words):' %depth)
            for span_length in sorted(depth_length_counts_sent[depth].keys()):
              print('        D%d length %d count = %d' %(depth, span_length, len(depth_length_counts_sent[depth][span_length])))
              for s in depth_length_counts_sent[depth][span_length]:
                print('          %s' %s)
          print('')
          print('  Num gold constituents: %d' %len(g_spans_complex))
          print('  Num test constituents: %d' %len(t_spans_complex))
          print('  Matching: %d' %len(brac_same))
          print('  False positives (in test but not gold): %d' %brac_t_only)
          print('  False negatives (in gold but not test): %d' %brac_g_only)
          print('  Precision: %.4f' %brac_p)
          print('  Recall: %.4f' %brac_r)
          print('  F-Measure: %.4f' %brac_f1)
          print('')
          print(debug_maxproj)
        
        sentid += 1

        # Read the next pair of trees
        g_line = gold.readline()
        while g_line and g_line.strip() == '':
          g_line = gold.readline()
        t_line = test.readline()
        while t_line and t_line.strip() == '':
          t_line = test.readline()

  # Get overall segmentation accuracy scores
  seg_p, seg_r, seg_f1 = accuracy(seg_tp, seg_fp, seg_fn)
  brac_p, brac_r, brac_f1 = accuracy(brac_tp, brac_fp, brac_fn)
  all_p, all_r, all_f1 = accuracy(all_tp, all_fp, all_fn)

  # Get part of speech accuracy scores
  pos_m2one = m21(leaves_t2g)
  pos_one2m = m21(leaves_g2t)
  pos_h = homog(leaves_t2g, leaves_g2t, n_wrds) 
  pos_c = homog(leaves_g2t, leaves_t2g, n_wrds)
  pos_vm = (2 * pos_h * pos_c) / (pos_h + pos_c)

  # Get constituent labeling accuracy scores
  lab_m2one = m21(same_t2g)
  lab_m2one = m21(same_g2t)
  lab_h = homog(same_t2g, same_g2t, brac_tp)
  lab_c = homog(same_g2t, same_t2g, brac_tp)
  lab_vm = (2 * lab_h * lab_c) / (lab_h + lab_c)

  if 'NP' in same_g2t:
    # Get NP prediction scores.
    # Start with overlapping bracketings...
    test_np = None
    test_np_count = 0
    total_same = 0
    test_np = max(t_cat_counts, key=lambda x: t_cat_counts[x])
    test_np_count = t_cat_counts[test_np]
    np_predict_tp = same_t2g[test_np]['NP']
    np_predict_fn = max_proj_scores['NP']['hit'] + max_proj_scores['NP']['miss'] - np_predict_tp
    np_predict_fp = t_cat_counts[test_np] - np_predict_tp
    np_predict_p, np_predict_r, np_predict_f1 = accuracy(np_predict_tp, np_predict_fp, np_predict_fn)

  # Print final evaluation scores
  print('')
  print('######################################')
  print('Corpus-wide eval results')
  print('######################################')
  print('')
  print('Minimum constituent length = %d' %args.minlength)
  print('')
  print('Overall (character spans, includes segmentation and bracketing):')
  print('  Precision: %.4f' %all_p)
  print('  Recall: %.4f' %all_r)
  print('  F-Measure: %.4f' %all_f1)
  print('')
  print('Segmentation:')
  print('  Precision: %.4f' %seg_p)
  print('  Recall: %.4f' %seg_r)
  print('  F-Measure: %.4f' %seg_f1)
  print('')
  print('Bracketing:')
  print('  Precision: %.4f' %brac_p)
  print('  Recall: %.4f' %brac_r)
  print('  F-Measure: %.4f' %brac_f1)
  print('')
  print('Tagging accuracy:')
  print('  Many-to-1 accuracy: %.4f' %pos_m2one)
  print('  1-to-many accuracy: %.4f' %pos_one2m)
  print('  Homogeneity: %.4f' %pos_h)
  print('  Completeness: %.4f' %pos_c)
  print('  V-Measure: %.4f' %pos_vm)
  print('')
  print('Labeling accuracy of correct constituents:')
  print('  Many-to-1 accuracy: %.4f' %lab_m2one)
  print('  1-to-many accuracy: %.4f' %lab_m2one)
  print('  Homogeneity: %.4f' %lab_h)
  print('  Completeness: %.4f' %lab_c)
  print('  V-Measure: %.4f' %lab_vm)
  print('')
  if 'NP' in same_g2t:
    print('NP prediction accuracy -- map most frequent test label ("%s") to "NP":' %test_np)
    print('  Precision: %.4f' %np_predict_p)
    print('  Recall: %.4f' %np_predict_r)
    print('  F-Measure: %.4f' %np_predict_f1)
    print('')
  print('Depth counts:')
  for depth in depth_counts:
    print('  Depth = %d: %d' %(depth, depth_counts[depth]))
    print('    Depth = %d counts by span length (words):' %depth)
    for span_length in sorted(depth_length_counts[depth].keys()):
      print('      D%d length %d count = %d' %(depth, span_length, depth_length_counts[depth][span_length]))
    print('')
  print('Maximal projection identification accuracy:')
  print('')
  for cat in max_proj:
    if max_proj_scores[cat]['hit'] + max_proj_scores[cat]['miss'] > 0:
      print('Type %s' %cat)
      print('  Found: %d' %max_proj_scores[cat]['hit'])
      print('  Not found: %d' %max_proj_scores[cat]['miss'])
      print('  % ' + cat + ' constituents identified: %.4f' %(float(max_proj_scores[cat]['hit']) / float(max_proj_scores[cat]['hit'] + max_proj_scores[cat]['miss'])))
      if cat in same_g2t:
        print('  Labeling of found ' + cat + ' constituents:')
        total = max_proj_scores[cat]['hit']
        for label in same_g2t[cat]:
          print('    Label ' + label + ': %.2f' %(float(same_g2t[cat][label])/float(total) * 100))
        print('')
    else:
      print('  No instances of %s found in gold trees' %cat)
      print('')

# Run main program
main()
