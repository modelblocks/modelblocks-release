import sys, re, argparse

argparser = argparse.ArgumentParser('''
Generates a space-delimited table of relevant fields from the entries in a *.constitevallist table.
''')
argparser.add_argument('constitevallist', type=str, nargs=1, help='Path to a *.constitevallist table.')
args, unknown = argparser.parse_known_args()

measure = re.compile(': *([0-9]+(\.[0-9]+)?)')
depth = re.compile('  Depth = ([0-9]+): ([0-9]+)')
type = re.compile(' *Type +([^ ]+)')
label = re.compile(' *Label +([^ :]+): ([0-9\.]+)')

with open(args.constitevallist[0], 'rb') as list:
  headers = list.readline().strip()
  print(headers + \
      ' allP allR allF1 ' + \
      'segP segR segF1 ' + \
      'bracP bracR bracF1 ' + \
      'mTo1Tag oneToMTag homogTag completTag vmTag ' + \
      'mTo1Lab oneToMLab homogLab completLab vmLab ' + \
      'np_predict_p np_predict_r np_predict_f1 ' + \
      ' '.join(['depth%d'%x for x in range(1,6)])),
  i = 0
  gLabs = []

  for f in list.readlines():
    row = f.strip()
    path = row.split(' ')[0]
    with open(path, 'rb') as f:
      line = f.readline()
      while line and line.strip() != '######################################':
	line = f.readline()
      line = f.readline()
      assert line.strip() == 'Corpus-wide eval results', 'Badly-formed input'
      line = f.readline()
      assert line.strip() == '######################################', 'Badly-formed input'

      line = f.readline()
      while line and not line.startswith('Overall'):
	line = f.readline()
      line = f.readline()
      assert line.startswith('  Precision:'), 'Badly-formed input'
      all_p = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  Recall:'), 'Badly-formed input'
      all_r = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  F-Measure:'), 'Badly-formed input'
      all_f1 = measure.search(line).group(1)

      line = f.readline()
      while line and not line.startswith('Segmentation'):
	line = f.readline()
      line = f.readline()
      assert line.startswith('  Precision:'), 'Badly-formed input'
      seg_p = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  Recall:'), 'Badly-formed input'
      seg_r = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  F-Measure:'), 'Badly-formed input'
      seg_f1 = measure.search(line).group(1)

      line = f.readline()
      while line and not line.startswith('Bracketing'):
	line = f.readline()
      line = f.readline()
      assert line.startswith('  Precision:'), 'Badly-formed input'
      brac_p = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  Recall:'), 'Badly-formed input'
      brac_r = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  F-Measure:'), 'Badly-formed input'
      brac_f1 = measure.search(line).group(1)

      line = f.readline()
      while line and not line.startswith('Tagging'):
	line = f.readline()
      line = f.readline()
      assert line.startswith('  Many-to-1 accuracy:'), 'Badly-formed input'
      mTo1Tag = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  1-to-many accuracy:'), 'Badly-formed input'
      oneToMTag = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  Homogeneity:'), 'Badly-formed input'
      homogTag = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  Completeness:'), 'Badly-formed input'
      completTag = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  V-Measure:'), 'Badly-formed input'
      vmTag = measure.search(line).group(1)

      line = f.readline()
      while line and not line.startswith('Labeling'):
	line = f.readline()
      line = f.readline()
      assert line.startswith('  Many-to-1 accuracy:'), 'Badly-formed input'
      mTo1Lab = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  1-to-many accuracy:'), 'Badly-formed input'
      oneToMLab = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  Homogeneity:'), 'Badly-formed input'
      homogLab = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  Completeness:'), 'Badly-formed input'
      completLab = measure.search(line).group(1)
      line = f.readline()
      assert line.startswith('  V-Measure:'), 'Badly-formed input'
      vmLab = measure.search(line).group(1)

      line = f.readline()
      while line and not (line.startswith('NP prediction') or line.startswith('Depth counts:')):
	line = f.readline()
      if line.startswith('NP prediction'):
        line = f.readline()
        assert line.startswith('  Precision:'), 'Badly-formed input'
        np_predict_p = measure.search(line).group(1)
        line = f.readline()
        assert line.startswith('  Recall:'), 'Badly-formed input'
        np_predict_r = measure.search(line).group(1)
        line = f.readline()
        assert line.startswith('  F-Measure:'), 'Badly-formed input'
        np_predict_f1 = measure.search(line).group(1)

        line = f.readline()
        while line and not line.startswith('Depth counts:'):
          line = f.readline()

      line = f.readline()
      depths = {1:0, 2:0, 3:0, 4:0, 5:0}
      while line and line.startswith('  Depth = '):
	d, count = depth.match(line).groups()
	depths[int(d)] = int(count)
	line = f.readline()

      proj_stats = {}

      while line:
	while line and not line.startswith('Type '):
	  line = f.readline()
        if line:
	  gLab = type.match(line.strip()).group(1)
	  if i == 0:
	    gLabs.append(gLab)
	  proj_stats[gLab] = {}
	  line = f.readline()
	  assert line.startswith('  Found:'), 'Badly-formed input'
	  found = measure.search(line.strip()).group(1)
	  proj_stats[gLab]['found'] = found
	  line = f.readline()
	  assert line.startswith('  Not found:'), 'Badly-formed input'
	  not_found = measure.search(line.strip()).group(1)
	  proj_stats[gLab]['not_found'] = not_found
	  line = f.readline()
	  assert line.startswith('  % '), 'Badly-formed input'
	  percent_ident = str(float(measure.search(line.strip()).group(1))*100)
	  proj_stats[gLab]['percent_ident'] = percent_ident
	  line = f.readline()
	  assert line.startswith('  Labeling of found '), 'Badly-formed input'
	  line = f.readline()
	  while line and (line.startswith('    Label ') or line.strip() == ''):
	    line = f.readline()
      
      if i == 0:
	newcols = ''
	for lab in gLabs:
	  newcols += '%sFound %sNotFound percent%sIdent ' %(lab, lab, lab)
	print(newcols)

      print(row),
      print(str(all_p) + ' ' + str(all_r) + ' ' + str(all_f1) + ' ' + \
            str(seg_p) + ' ' + str(seg_r) + ' ' + str(seg_f1) + ' ' + \
            str(brac_p) + ' ' + str(brac_r) + ' ' + str(brac_f1) + ' ' + \
	    str(mTo1Tag) + ' ' + str(oneToMTag) + ' ' + str(homogTag) + ' ' + str(completTag) + ' ' + str(vmTag) + ' ' + \
	    str(mTo1Lab) + ' ' + str(oneToMLab) + ' ' + str(homogLab) + ' ' + str(completLab) + ' ' + str(vmLab) + ' ' + \
	    ' '.join([str(depths[x]) for x in range(1,6)]) + ' ' + \
	    ' '.join(['%s %s %s' %(proj_stats[x]['found'], proj_stats[x]['not_found'], proj_stats[x]['percent_ident']) for x in gLabs]))

      i += 1
