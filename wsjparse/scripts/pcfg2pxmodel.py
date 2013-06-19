
#### create branch- and depth-specific pcfg

import sys
import model
import re

MAX_ITER = 20

#### 0) read model from stdin
mCr = model.Model('Cr')
mCC = model.CondModel('CC')
mCu = model.CondModel('Cu')
mX  = model.CondModel('X')
mPc = model.CondModel('Pc')
mPw = model.CondModel('Pw')
mP = model.Model('P')
mW = model.Model('W')
for s in sys.stdin:
    mCr.read(s)
    mCC.read(s)
    mCu.read(s)
    mX.read(s)
    mPc.read(s)
    mPw.read(s)
    mP.read(s)
    mW.read(s)
# add terminal rules (comment out if terminal and nonterminal categories intersect)
#for c in mX:
#    mCC[c]['-','-'] = 1.0

#mCr.normalize()
#mCu.normalize()
#mCC.normalize()

# immediately dump and clear all models that we won't need, to save memory
mPw.write()
mPw.clear()
mW.write()
mW.clear()
mP.write()
mP.clear()

sys.stderr.write("0 - read in and dumped unneeded models\n")

#### 1) obtain closure of unary rules
mCu_curr = model.CondModel('Cu')
mCu_star = model.CondModel('Cu*')
# init
for c in mCC:
    mCu_curr[c][c] = 1.0
    mCu_star[c][c] = 1.0
for c in mX:
    mCu_curr[c][c] = 1.0
    mCu_star[c][c] = 1.0
#mCu_curr['-']['-'] = 1.0
#mCu_star['-']['-'] = 1.0
# for each derivation length
for k in range(1,MAX_ITER):
    mCu_prev = mCu_curr
    mCu_curr = model.CondModel('Cu')
    for c in mCu_prev:
        for ch in mCu_prev[c]:
            for ch0 in mCu.get(ch):
                mCu_curr[c][ch0] += mCu_prev[c][ch] * mCu[ch][ch0]
                mCu_star[c][ch0] += mCu_prev[c][ch] * mCu[ch][ch0]

sys.stderr.write("1 - calculated closure of unary rules\n")


#### 2) obtain side- and depth-specific preterminal unary rules

## 2a) cross unary closure with preterminal unary rules
mX_cnf  = model.CondModel('X_cnf')
for c in mCu_star:
    for ch in mCu_star[c]:
        for x in mX[ch]:
            mX_cnf[c][x] += mCu_star[c][ch] * mX[ch][x]
mX.clear()

## 2b) make preterminal unary rules branch and depth specific
# for Pc
mPc_new = model.CondModel('Pc')
for s in ['L','R']:
    for d in [1,2,3,4,5]:
        for c in mPc:
            for p in mPc[c]:
                mPc_new[c+'^'+s+','+str(d)][p] = mPc[c][p]
mPc.clear()
# for X
mPX = model.CondModel('PX')
mX_new = model.CondModel('X')
for c in mX_cnf:
	for x in mX_cnf[c]:
		mX_new[c][x] = mX_cnf[c][x]
		for s in ['L','R']:
			for d in [1,2,3,4,5]:
				mPX[c+'^'+s+','+str(d)][c] = 1.0
		mX_new.write()
		mX_new.clear()
mPX.write()
mPX.clear()

mX_cnf.clear()

## 2c) print terminal unary rules
mPc_new.write()
mPc_new.clear()
#mX_new.write()
#mX_new.clear()

sys.stderr.write("2 - calculated side- and depth-specific preterminal unary rules\n")


#### 3) obtain side- and depth-specific nonterminal binary rules

## 3a) cross unary closure with nonterminal binary rules
mCC_cnf = model.CondModel('CC_cnf')
for c in mCu_star:
    for ch in mCu_star[c]:
        for ch0,ch1 in mCC[ch]:
            mCC_cnf[c][ch0,ch1] += mCu_star[c][ch] * mCC[ch][ch0,ch1]
mCC.clear()
#for c in mCC:
#    for c0,c1 in mCC[c]:
#        for c0h in mCu_star[c0]:
#            for c1h in mCu_star[c1]:
#                mCC_cnf[c][c0h,c1h] += mCC[c][c0,c1] * mCu_star[c0][c0h] * mCu_star[c1][c1h]
#mCC_cnf.write()

EB = {}
EE = {}
for c in mCC_cnf:
    EB[c] = True if re.search('-eb',c) is not None else False
    for c0,c1 in mCC_cnf[c]:
        EE[c1] = True if re.search('-ee',c1) is not None else False

## 3b) compute prob that a subtree below constit will fit at each branch and depth
mT_curr = model.Model('T')
# for each derivation length
for k in range(1,MAX_ITER):
    sys.stderr.write('  '+str(k)+'\n')
    mT_prev = mT_curr
    mT_curr = model.Model('T')
    # init null items in sd model (assume we have 'CC x : - - = p' for terminals; otherwise just init terms)
    for s in ['L','R']:
        for d in range(1,6):
            mT_curr['-',s,d] = 1.0
    # fill in rest of sd model
    for d in range(1,5):
        for c in mCC_cnf:
            for c0,c1 in mCC_cnf[c]:
                if not EB[c] or (not EE[c] and not EE[c1]):
                    mT_curr[c,'L',d] += mCC_cnf[c][c0,c1] * mT_prev[c0,'L',d  ] * mT_prev[c1,'R',d]
                if EB[c] and not EE[c] and not EE[c1]:
                    mT_curr[c,'R',d] += mCC_cnf[c][c0,c1] * mT_prev[c,'L',d+1]
                elif EE[c1] and not EB[c] and not EB[c1]:
                    mT_curr[c,'R',d] += mCC_cnf[c][c0,c1] * mT_prev[c0,'L',d+1] * mT_prev[c1,'R',d-1]
                else:
                    mT_curr[c,'R',d] += mCC_cnf[c][c0,c1] * mT_prev[c0,'L',d+1] * mT_prev[c1,'R',d]
#mT_curr.write()
# NOTE: don't normalize, since mT stores fraction of prob mass fitting under d

## 3c) renormalize over prob mass that fits
mCC_new = model.CondModel('CC')
for d in range(1,5):
    for c in mCC_cnf:
        for c0,c1 in mCC_cnf[c]:
            if c0=='-' and c1=='-':
                mCC_new[c+'^L,'+str(d)][c0,c1] += mCC_cnf[c][c0,c1] * mT_curr[c0,'L',d  ] * mT_curr[c1,'R',d]
                mCC_new[c+'^R,'+str(d)][c0,c1] += mCC_cnf[c][c0,c1] * mT_curr[c0,'L',d+1] * mT_curr[c1,'R',d]
            else:
                if not EB[c] or (not EE[c] and not EE[c1]):
                    mCC_new[c+'^L,'+str(d)][c0+'^L,'+str(d),  c1+'^R,'+str(d)] += mCC_cnf[c][c0,c1] * mT_curr[c0,'L',d  ] * mT_curr[c1,'R',d]
                if EB[c] and not EE[c] and not EE[c1]:
                    mCC_new[c+'^R,'+str(d)][c+'^L,'+str(d+1),'-'+'^R,'+str(d)] += mCC_cnf[c][c0,c1] * mT_curr[c,'L',d+1]
                elif EE[c1] and not EB[c] and not EB[c1]:
                    mCC_new[c+'^R,'+str(d)][c0+'^L,'+str(d+1),c1+'^R,'+str(d)] += mCC_cnf[c][c0,c1] * mT_curr[c0,'L',d+1] * mT_curr[c1,'R',d-1]
                else:
                    mCC_new[c+'^R,'+str(d)][c0+'^L,'+str(d+1),c1+'^R,'+str(d)] += mCC_cnf[c][c0,c1] * mT_curr[c0,'L',d+1] * mT_curr[c1,'R',d]
mCC_new.normalize()
mCC_new.write()

sys.stderr.write("3 - calculated side- and depth-specific nonterminal binary rules\n")


#### 4) obtain branch- and depth-specific prior model: Cr

mCr_new = model.Model('Cr')
for c in mCr:
    mCr_new[c+'^L,1'] = mCr[c]
mCr_new.write()

sys.stderr.write("4 - calculated side- and depth-specific prior Cr\n")
