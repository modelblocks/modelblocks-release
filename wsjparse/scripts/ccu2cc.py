
#### create branch- and depth-specific pcfg

import sys
import model

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
for c in mX:
    mCC[c]['-','-'] = 1.0

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
for c in mCu:
	mCu_curr[c][c] = 1.0
	mCu_star[c][c] = 1.0
for c in mCC:
    mCu_curr[c][c] = 1.0
    mCu_star[c][c] = 1.0
for c in mX:
    mCu_curr[c][c] = 1.0
    mCu_star[c][c] = 1.0
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


# 2) cross unary closure with preterminal unary rules
mX_cnf  = model.CondModel('X')
for c in mCu_star:
    for ch in mCu_star[c]:
        for x in mX[ch]:
            mX_cnf[c][x] += mCu_star[c][ch] * mX[ch][x]
mX_cnf.write()
mX.clear()

sys.stderr.write("2 - crossed unaries with preterminal unary rules\n")


# 3) cross unary closure with nonterminal binary rules
mCC_cnf = model.CondModel('CC')
for c in mCu_star:
    for ch in mCu_star[c]:
        for ch0,ch1 in mCC[ch]:
            mCC_cnf[c][ch0,ch1] += mCu_star[c][ch] * mCC[ch][ch0,ch1]
mCC_cnf.write()
mCC.clear()

sys.stderr.write("3 - crossed unaries with nonterminal binary rules\n")


# 4) write prior model
mCr.write()

