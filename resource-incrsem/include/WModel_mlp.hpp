////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. //
//                                                                            //
//  ModelBlocks is free software: you can redistribute it and/or modify       //
//  it under the terms of the GNU General Public License as published by      //
//  the Free Software Foundation, either version 3 of the License, or         //
//  (at your option) any later version.                                       //
//                                                                            //
//  ModelBlocks is distributed in the hope that it will be useful,            //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of            //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
//  GNU General Public License for more details.                              //
//                                                                            //
//  You should have received a copy of the GNU General Public License         //
//  along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

//#include <typeinfo>
//#include <regex>
//#include <algorithm>

// TODO: have parser identify this automatically
// XModel E, K, P, Char, RNN hidden sizes
const uint X_E_SIZE = 20;
const uint X_K_SIZE = 400;
const uint X_P_SIZE = 20;
const uint X_C_SIZE = 20;
const uint X_H_SIZE = 460;

// MModel E, P, LCat, Char, RNN hidden sizes
const uint M_E_SIZE = 20;
const uint M_P_SIZE = 20;
const uint M_L_SIZE = 400;
const uint M_C_SIZE = 20;
const uint M_H_SIZE = 460;

// maybe include as part of WModel
vector<string> PUNCT = { "-LCB-", "-LRB-", "-RCB-", "-RRB-" };

////////////////////////////////////////////////////////////////////////////////

class WPredictor : public DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX> {
  public:
    WPredictor ( ) : DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX>(){}
    WPredictor ( EVar e, K k, CVar c ) : DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX>(e,k,c){}
};


class WModel {

  public:
    typedef DelimitedTrip<psX,Delimited<EVar>,psPipe,Delimited<K>,psPipe,Delimited<CVar>,psX> WPredictor;
    typedef DelimitedTrip<psX,Delimited<EVar>,psPipe,Delimited<CVar>,psPipe,Delimited<string>,psX> MPredictor;
    typedef DelimitedCol<psLBrack, double, psComma, psRBrack> DenseVec;

  private:

    // map from pair<lemma, primcat> to list of compatible WPredictors and MPredictors (read in from WModel)
    map<pair<string,string>,DelimitedList<psLBrack,WPredictor,psSpace,psRBrack>> mxwp;
    map<pair<string,string>,DelimitedList<psLBrack,MPredictor,psSpace,psRBrack>> mxmp;

    // map from predictor components to dense vectors
    // XModel (E, K, P, char to dense vector)
    map<EVar,vec> mxev;
    map<K,vec> mxkv;
    map<CVar,vec> mxpv;
    map<string,vec> mxcv;
    // MModel (E, P, SK, char to dense vector)
    map<EVar,vec> mmev;
    map<CVar,vec> mmpv;
    map<string,vec> mmlv;
    map<string,vec> mmcv;

    map<string,unsigned int> mci; // map from character to index (required for indexing probabilities)
    map<string,unsigned int> mmi; // map from morph rule to index (required for indexing probabilities)

    // weights and biases for XModel
    DelimitedVector<psX, double, psComma, psX> xihw;  // SRN i2h
    DelimitedVector<psX, double, psComma, psX> xhhw;  // SRN h2h
    DelimitedVector<psX, double, psComma, psX> xfcw;  // FC classifier
    DelimitedVector<psX, double, psComma, psX> xihb;  // SRN i2h bias
    DelimitedVector<psX, double, psComma, psX> xhhb;  // SRN h2h bias
    DelimitedVector<psX, double, psComma, psX> xfcb;  // FC classifier bias
    mat xihwm;
    mat xhhwm;
    mat xfcwm;
    vec xihbv;
    vec xhhbv;
    vec xfcbv;

    // weights and biases for MModel
    DelimitedVector<psX, double, psComma, psX> mihw;  // SRN i2h
    DelimitedVector<psX, double, psComma, psX> mhhw;  // SRN h2h
    DelimitedVector<psX, double, psComma, psX> mfcw;  // FC classifier
    DelimitedVector<psX, double, psComma, psX> mihb;  // SRN i2h bias
    DelimitedVector<psX, double, psComma, psX> mhhb;  // SRN h2h bias
    DelimitedVector<psX, double, psComma, psX> mfcb;  // FC classifier bias
    mat mihwm;
    mat mhhwm;
    mat mfcwm;
    vec mihbv;
    vec mhhbv;
    vec mfcbv;

  public:

    // map from W to map from WPredictor to P(W | WPredictor)
    typedef map<WPredictor,double> WPPMap;
    typedef map<W,WPPMap> WWPPMap;
    // map from pair<lemma, primcat> to vector of P(lemma | WPredictor)
    typedef map<pair<string,string>,rowvec> XPMap;
    // map from pair<lemma, primcat> to matrix of P(M | WPredictor lemma)
    typedef map<pair<string,string>,mat> MPMap;

    WModel ( ) { }
    WModel ( istream& is ) {
      while( is.peek()=='W' ) {
        Delimited<char> i;
        Delimited<char> j;
        is >> "W " >> i >> " " >> j >> " ";
        if (i == 'X') {
          if (j == 'I') is >> xihw >> "\n";
          if (j == 'i') is >> xihb >> "\n";
          if (j == 'H') is >> xhhw >> "\n";
          if (j == 'h') is >> xhhb >> "\n";
          if (j == 'F') is >> xfcw >> "\n";
          if (j == 'f') is >> xfcb >> "\n";
        }
        else if (i == 'M') {
          if (j == 'I') is >> mihw >> "\n";
          if (j == 'i') is >> mihb >> "\n";
          if (j == 'H') is >> mhhw >> "\n";
          if (j == 'h') is >> mhhb >> "\n";
          if (j == 'F') is >> mfcw >> "\n";
          if (j == 'f') is >> mfcb >> "\n";
        }
      }
      //cerr << "done with Ws" << endl;
      while ( is.peek()=='E' ) {
        Delimited<char> i;
        Delimited<EVar> e;
        is >> "E " >> i >> " " >> e >> " ";
        if (i == 'X') {
          DenseVec dv = DenseVec(X_E_SIZE);
          is >> dv >> "\n";
          mxev.try_emplace(e, vec(dv));
        }
        else if (i == 'M') {
          DenseVec dv = DenseVec(M_E_SIZE);
          is >> dv >> "\n";
          mmev.try_emplace(e, vec(dv));
        }
      }
      //cerr << "done with Es" << endl;
      while ( is.peek()=='K' ) {
        Delimited<K> k;
        DenseVec dv = DenseVec(X_K_SIZE);
        is >> "K " >> k >> " " >> dv >> "\n";
        mxkv.try_emplace(k, vec(dv));
      }
      //cerr << "done with Ks" << endl;
      while ( is.peek()=='P' ) {
        Delimited<char> i;
        Delimited<CVar> c;
        is >> "P " >> i >> " " >> c >> " ";
        if (i == 'X') {
          DenseVec dv = DenseVec(X_P_SIZE);
          is >> dv >> "\n";
          mxpv.try_emplace(c, vec(dv));
        }
        else if (i == 'M') {
          DenseVec dv = DenseVec(M_P_SIZE);
          is >> dv >> "\n";
          mmpv.try_emplace(c, vec(dv));
        }
      }
      //cerr << "done with Ps" << endl;
      while ( is.peek()=='L' ) {
        string l;
        DenseVec dv = DenseVec(M_L_SIZE);
        is >> "L " >> l >> " " >> dv >> "\n";
        mmlv.try_emplace(l, vec(dv));
      }
      //cerr << "done with Ls" << endl;
      while ( is.peek()=='C' ) {
        Delimited<char> i;
        string c;
        is >> "C " >> i >> " " >> c >> " ";
        if (i == 'X') {
          DenseVec dv = DenseVec(X_C_SIZE);
          is >> dv >> "\n";
          mxcv.try_emplace(c, vec(dv));
        }
        else if (i == 'M') {
          DenseVec dv = DenseVec(M_C_SIZE);
          is >> dv >> "\n";
          mmcv.try_emplace(c, vec(dv));
        }
        else if (i == 'I') {
          is >> mci[c] >> "\n";
        }
      }
      //cerr << "done with Cs" << endl;
      while ( is.peek()=='R' ) {
        string x;
        is >> "R " >> x >> " ";
        is >> mmi[x] >> "\n";
      }
      //cerr << "done with Rs" << endl;
      while ( is.peek()=='X' ) {
        string x;
        string p;
        DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> wp;
        is >> "X " >> x >> " " >> p >> " " >> wp >> "\n";
        pair<string,string> xppair (x,p);
        mxwp.try_emplace(xppair, wp);
      }
      //cerr << "done with Xs" << endl;
      while ( is.peek()=='M' ) {
        string x;
        string p;
        DelimitedList<psLBrack,MPredictor,psSpace,psRBrack> mp;
        is >> "M " >> x >> " " >> p >> " " >> mp >> "\n";
        pair<string,string> xppair (x,p);
        mxmp.try_emplace(xppair, mp);
      }
      //cerr << "done with Ms" << endl;
      // initialize armadillo mat/vecs
      xihwm = xihw;
      xhhwm = xhhw;
      xfcwm = xfcw;
      xihbv = xihb;
      xhhbv = xhhb;
      xfcbv = xfcb;
      xihwm.reshape(X_H_SIZE, X_E_SIZE + X_K_SIZE + X_P_SIZE + X_C_SIZE);
      xhhwm.reshape(X_H_SIZE, X_H_SIZE);
      xfcwm.reshape(xfcw.size()/X_H_SIZE, X_H_SIZE);

      mihwm = mihw;
      mhhwm = mhhw;
      mfcwm = mfcw;
      mihbv = mihb;
      mhhbv = mhhb;
      mfcbv = mfcb;
      mihwm.reshape(M_H_SIZE, M_E_SIZE + M_P_SIZE + M_L_SIZE + M_C_SIZE);
      mhhwm.reshape(M_H_SIZE, M_H_SIZE);
      mfcwm.reshape(mfcw.size()/M_H_SIZE, M_H_SIZE);

    }

    // XModel: index input character embedding
    const mat getXCharMat( string a, unsigned int i ) const {
      auto it = mxcv.find( a );
      assert ( it != mxcv.end() );
      return repmat(it->second, 1, i);
    }

    string removeUnkChar( string s ) const {
      unsigned i = 0;
      while ( i < s.length() ){
        string str(1, s[i]);
        //cerr << "Processing character " << s[i] << "." << endl;
        if ( mci.find(str) == mci.end() ) {
          cerr << "Unknown character " << s[i] << " found in " << s << "." << endl;
          s.erase(std::remove(s.begin(), s.end(), s[i]), s.end());
        } else {
          i++;
        }
      }
      return s;
    }

    // XModel: index list of compatible WPredictors given pair<lemma, primcat>
    const DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> getWPredictorList( pair<string,string> xsp ) const {
      if ( isdigit(xsp.first.at(0)) ) {
        pair<string,string> numpair ("NUM", "All");
        return mxwp.find(numpair)->second;
      } else {
        auto it = mxwp.find( xsp );
        pair<string,string> unkpair ("UNK", xsp.second);
        auto unklist = mxwp.find(unkpair)->second;
        if ( it != mxwp.end() ) {
          auto predlist = it->second;
          unklist.splice(unklist.begin(), predlist);
          return unklist;
        } else {
          return unklist;
        }
      }
    }

    // XModel: index input WPredictor embedding given list of WPredictors
    const mat getWPredictorMat( DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> lwp ) const {
      unsigned int idx = 0;
      mat wpmat = mat(X_E_SIZE + X_K_SIZE + X_P_SIZE, lwp.size());
      for ( auto& wp : lwp ) {
        auto ite = mxev.find( wp.first() );
        assert ( ite != mxev.end() );
        auto itk = mxkv.find( wp.second() );
        assert ( itk != mxkv.end() );
        auto itp = mxpv.find( wp.third() );
        assert ( itp != mxpv.end() );
        if( itp!=mxpv.end() ) {
          wpmat.col(idx) = join_cols(join_cols(ite->second, itk->second), (itp==mxpv.end()) ? zeros(X_P_SIZE) : itp->second);
          idx ++;
        }
      }
      return wpmat;
    }

    // XModel: index input character index
    const unsigned int getXCharIndex( string a ) const {
      auto it = mci.find( a );
      assert ( it != mci.end() );
      return it->second;
    }

    // MModel: index input character embedding
    const mat getMCharMat( string a, unsigned int i ) const {
      auto it = mmcv.find( a );
      assert ( it != mmcv.end() );
      return repmat(it->second, 1, i);
    }

    // MModel: index list of compatible MPredictors given pair<lemma, primcat>
    const DelimitedList<psLBrack,MPredictor,psSpace,psRBrack> getMPredictorList( pair<string,string> xsp ) const {
      if ( isdigit(xsp.first.at(0)) ) {
        pair<string,string> numpair ("NUM", "All");
        return mxmp.find(numpair)->second;
      } else {
        auto it = mxmp.find( xsp );
        pair<string,string> unkpair ("UNK", xsp.second);
        auto unklist = mxmp.find(unkpair)->second;
        if ( it != mxmp.end() ) {
          auto predlist = it->second;
          unklist.splice(unklist.begin(), predlist);
          return unklist;
        } else {
          return unklist;
        }
      }
    }

    // MModel: index input MPredictor embedding given list of MPredictors
    const mat getMPredictorMat( DelimitedList<psLBrack,MPredictor,psSpace,psRBrack> lmp ) const {
      unsigned int idx = 0;
      mat mpmat = mat(M_E_SIZE + M_P_SIZE + M_L_SIZE, lmp.size());
      for ( auto& mp : lmp ) {
        auto ite = mmev.find( mp.first() );
        assert ( ite != mmev.end() );
        auto itp = mmpv.find( mp.second() );
        assert ( itp != mmpv.end() );
        auto itl = mmlv.find( mp.third() );
        assert ( itl != mmlv.end() );
        mpmat.col(idx) = join_cols(join_cols(ite->second, itp->second), itl->second);
        idx ++;
      }
      return mpmat;
    }

    // MModel: index morph rule index
    const unsigned int getMRuleIndex( string a ) const {
      auto it = mmi.find( a );
      assert ( it != mmi.end() );
      return it->second;
    }

    // takes input word and iterates over morph rules that are read in as part of the WModel
    // if morph rule can apply, generates <<lemma, primcat>, rule> and appends it to list
    const list<pair<pair<string,string>,string>> applyMorphRules ( const W& w_t ) const {
      list<pair<pair<string,string>,string>> lxmp;
      string sW = w_t.getString().c_str();

      // do not lowercase word if special punctuation token ("-LCB-", "-LRB-", "-RCB-", "-RRB-")
      if ( find( PUNCT.begin(), PUNCT.end(), sW ) == PUNCT.end() ) {
        transform(sW.begin(), sW.end(), sW.begin(), [](unsigned char c) { return std::tolower(c); });
        sW = removeUnkChar(sW);
      }

      // loop over morph rules
      for ( const auto& mi : mmi ) {
        smatch mM;
        string sX;
        string sP;

        // for identity or annihilator rules, return the word itself as lemma
        if ( mi.first == "%|%" || mi.first == "%|" ) {
          sX = sW;
          sP = "All";
          if (!sX.empty()) lxmp.push_back(pair<pair<string,string>,string>(pair<string,string>(sX,sP),mi.first));
        } else {
          // otherwise, apply morph rule for lemma and primcat
          if ( regex_match( mi.first, mM, regex("^(.*)[%](.*)[|](.*)[%](.*)$") ) ) {
            smatch mW;
            if ( regex_match(sW, mW, regex("^(.*)"+string(mM[2])+"$")) ) {
              sX = string(mW[1])+string(mM[4]);
              sP = string(mM[3]);
              if (!sX.empty()) lxmp.push_back(pair<pair<string,string>,string>(pair<string,string>(sX,sP),mi.first));
            }
          }
        }
      }
      return lxmp;
    }

    // XModel: calculate P(lemma | WPredictor)
    // takes input pair<lemma, primcat> and calculates RNN probabilities
    rowvec calcLemmaLikelihoods( const pair<string,string>& xsp, XPMap& xpmap ) const {
      auto it = xpmap.find( xsp );
      rowvec seqlogprobs;
      if ( it == xpmap.end() ) {
        // index list of compatible WPredictors
        auto wplist = getWPredictorList( xsp );
        string x_t = xsp.first;
        seqlogprobs = zeros<mat>(1, wplist.size());
        mat xihbm = repmat(xihbv, 1, wplist.size());
        mat xhhbm = repmat(xhhbv, 1, wplist.size());
        mat xfcbm = repmat(xfcbv, 1, wplist.size());
        mat wpmat = getWPredictorMat(wplist);
        // calculate first hidden state with start character <S>
        mat ht = relu(xihwm * join_cols(wpmat, getXCharMat("<S>", wplist.size())) + xihbm + xhhbm);
        mat st_scores = exp(xfcwm * ht + xfcbm);
        rowvec st_norm = sum(st_scores, 0);
        mat st_logprobs = log(st_scores.each_row() / st_norm);
        rowvec st1_logprobs;


        if ( find( PUNCT.begin(), PUNCT.end(), x_t ) != PUNCT.end() ) {
          // if lemma is special punctuation token ("-LCB-", "-LRB-", "-RCB-", "-RRB-"), index probability using the token itself
          seqlogprobs += st_logprobs.row( getXCharIndex( x_t ) );
          mat ht1 = relu(xihwm * join_cols(wpmat, getXCharMat(x_t, wplist.size())) + xihbm + xhhwm * ht + xhhbm);
          mat st1_scores = exp(xfcwm * ht1 + xfcbm);
          rowvec st1_norm = sum(st1_scores, 0);
          st1_logprobs = log(st1_scores.row(getXCharIndex( "<E>" )) / st1_norm);
          seqlogprobs += st1_logprobs;
        } else {
              string c0(1, x_t[0]);
              // index probability for first character
              seqlogprobs += st_logprobs.row( getXCharIndex( c0.c_str() ));

              for ( unsigned i = 0; i < x_t.length(); ++i ){
                string ct(1, x_t[i]);
                string ct1(1, x_t[i+1]);
                mat ht1 = relu(xihwm * join_cols(wpmat, getXCharMat(ct.c_str(), wplist.size())) + xihbm + xhhwm * ht + xhhbm);
                mat st1_scores = exp(xfcwm * ht1 + xfcbm);
                rowvec st1_norm = sum(st1_scores, 0);

                if (i != x_t.length()-1) {
                  st1_logprobs = log(st1_scores.row(getXCharIndex( ct1.c_str() )) / st1_norm);
                } else {
                  st1_logprobs = log(st1_scores.row(getXCharIndex( "<E>" )) / st1_norm);
                }
                seqlogprobs += st1_logprobs;
                ht = ht1;
              }
            }
        xpmap.try_emplace(xsp, seqlogprobs);
        } else {
          seqlogprobs = it->second;
        }
      return seqlogprobs;
    }

    // MModel: calculate P(M | WPredictor lemma)
    // takes input pair<lemma, primcat> and calculates RNN probabilities
    mat calcRuleLikelihoods( const pair<string,string>& xsp, MPMap& mpmap ) const {
      auto it = mpmap.find( xsp );
      mat rulelogprobs;
      if ( it == mpmap.end() ) {
        // index list of compatible MPredictors
        auto mplist = getMPredictorList( xsp );
        string x_t = xsp.first;
        mat mihbm = repmat(mihbv, 1, mplist.size());
        mat mhhbm = repmat(mhhbv, 1, mplist.size());
        mat mfcbm = repmat(mfcbv, 1, mplist.size());
        mat mpmat = getMPredictorMat(mplist);

        if ( find( PUNCT.begin(), PUNCT.end(), x_t ) != PUNCT.end() ) {
          // if lemma is special punctuation token ("-LCB-", "-LRB-", "-RCB-", "-RRB-"), calculate probability using the token itself
          mat ht = relu(mihwm * join_cols(mpmat, getMCharMat(x_t, mplist.size())) + mihbm + mhhbm);
          mat st_scores = exp(mfcwm * ht + mfcbm);
          rowvec st_norm = sum(st_scores, 0);
          rulelogprobs = log(st_scores.each_row() / st_norm);
        } else {
            string c0(1, x_t[0]);
            // calculate probability using first character
//          cerr << "word " << x_t << " c0 " << c0 << endl;
            mat ht = relu(mihwm * join_cols(mpmat, getMCharMat(c0.c_str(), mplist.size())) + mihbm + mhhbm);

            for ( unsigned i = 1; i < x_t.length(); ++i ){
              string ct(1, x_t[i]);
//            cerr << "word " << x_t << " c" << i << " " << ct << endl;
              mat ht1 = relu(mihwm * join_cols(mpmat, getMCharMat(ct.c_str(), mplist.size())) + mihbm + mhhwm * ht + mhhbm);
              ht = ht1;
            }
            mat st_scores = exp(mfcwm * ht + mfcbm);
            rowvec st_norm = sum(st_scores, 0);
            rulelogprobs = log(st_scores.each_row() / st_norm);
          }
          mpmap.try_emplace(xsp, rulelogprobs);
      } else {
        rulelogprobs = it->second;
      }
      return rulelogprobs;
    }

    void calcPredictorLikelihoods( const W& w_t, const WWPPMap& wwppmap, XPMap& xpmap, MPMap& mpmap, WPPMap& wppmap ) const {
      auto it = wwppmap.find( w_t );
      if ( it == wwppmap.end() ) {
        // generate list of <<lemma, primcat>, rule>
        //cerr << "applying morph rules for word: " << w_t << endl;
        list<pair<pair<string,string>,string>> lxmp = applyMorphRules(w_t);
        // loop over <<lemma, primcat>, rule>
        for ( const auto& xmp : lxmp ) {
        //cerr << "generated word " << w_t << " from lemma " << xmp.first.first << ", primcat " << xmp.first.second << ", rule " << xmp.second << endl;
          DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> lwp = getWPredictorList(xmp.first);
          rowvec xll = calcLemmaLikelihoods(xmp.first, xpmap);
          mat mllall = calcRuleLikelihoods(xmp.first, mpmap);
          rowvec mll = mllall.row(getMRuleIndex(xmp.second));
          rowvec wprobs = exp(xll + mll);
          unsigned int idx = 0;
          for ( const auto& wp : lwp ) {
            wppmap[wp] += wprobs(idx);
            idx ++;
          }
        }
      } else {
        wppmap = it->second;
      }
    }
};

