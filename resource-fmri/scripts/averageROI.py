import sys

header = True
ROI = sys.argv[1:]
if 'AllROI' in ROI:
    ROI.pop(ROI.index('AllROI'))
for i in range(len(ROI)):
    ROI[i] = 'bold' + ROI[i]
ROI_cols = []

for line in sys.stdin:
    row = line.strip().split()
    if header:
        for h in row:
            if h in ROI:
                ROI_cols.append(row.index(h))
        header = False
        row.append('boldAllROI')
    else:
        bold = 0
        for c in ROI_cols:
            bold += float(row[c])
        bold /= len(ROI_cols)
        row.append(str(bold))
    print(' '.join(row))
