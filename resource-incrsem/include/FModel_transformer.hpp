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


class FPredictorVec {

  private:
    const BeamElement<HiddState>& be;
    const HVec& hvA;
    bool nullA;

  public:
    FPredictorVec( const BeamElement<HiddState>& belement, const HVec& hvAnt, bool nullAnt )
      : be (belement),
      hvA ((hvAnt.size() > 0) ? hvAnt : hvBot),
      nullA (nullAnt) 
    {
    }

    bool getNullA() {
      return nullA;
    }

    const HVec getHvA() {
      return hvA; //antecedent
    }

    const BeamElement<HiddState>& getBeamElement() const {
        return be;
    }

    friend ostream& operator<< ( ostream& os, const FPredictorVec& fpv ) {
      //const StoreState ss = fpv.getBeamElement().getHidd().getStoreState();
      const int d = getDepth(fpv.be);
      const CVar catBase = getCatBase(fpv.be);
      const HVec hvB = getHvB(fpv.be);
      const HVec hvF = getHvF(fpv.be);
        
      os << d << " " << catBase << " " << hvB << " " << hvF << " " << fpv.hvA << " " << fpv.nullA;
      return os;
    }
};



////////////////////////////////////////////////////////////////////////////////
class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> KDenseVec;

  private:
    static const uint FSEM_DIM_DEFAULT = 20;
    static const uint FSYN_DIM_DEFAULT = 20;
    static const uint FANT_DIM_DEFAULT = 20;
    uint FSEM_DIM;
    uint FSYN_DIM;
    uint FANT_DIM;
    uint num_heads;
    uint head_dim;
    uint attn_dim;

    map<CVar,vec> mcbv;                        // map between syntactic category and embeds
    map<KVec,vec> mkbdv;                  // map between KVec and embeds
    map<KVec,vec> mkfdv;
    map<KVec,vec> mkadv;

    map<FEK,unsigned int> mfeki;                // response indices
    map<unsigned int,FEK> mifek;

    unsigned int iNextResponse  = 0;

    // weights
    DelimitedVector<psX, double, psComma, psX> fwp; // pre-attention feedforward
    vector<vector<double>> fwi; // attention input projection (concatenation of 
    vector<vector<double>> fwo; // attention output projection for each transformer layer
    vector<vector<double>> fwf; // feedforward for each transformer layer
//    DelimitedVector<psX, double, psComma, psX> fwi; // attention input projection -- contains query, key, and value matrices
//    DelimitedVector<psX, double, psComma, psX> fwo; // attention output projection
//    DelimitedVector<psX, double, psComma, psX> fwf; // first feedforward
    DelimitedVector<psX, double, psComma, psX> fws; // second feedforward

    mat fwpm;
    vector<mat> fwim;
    vector<mat> fwqm; // query
    vector<mat> fwkm; // key
    vector<mat> fwvm; // value
    vector<mat> fwom;
    vector<mat> fwfm;
    mat fwsm;

    // biases
    DelimitedVector<psX, double, psComma, psX> fbp; // pre-attention feedforward
    vector<vector<double>> fbi; // attention input projection for each transformer layer
    vector<vector<double>> fbo; // attention output projection for each transformer layer
    vector<vector<double>> fbf; // feedforward for each transformer layer
//    DelimitedVector<psX, double, psComma, psX> fbi; // attention input projection
//    DelimitedVector<psX, double, psComma, psX> fbo; // attention output projection
//    DelimitedVector<psX, double, psComma, psX> fbf; // first feedforward
    DelimitedVector<psX, double, psComma, psX> fbs; // second feedforward

    vec fbpv;
    vector<vec> fbiv;
    vector<vec> fbqv; // query
    vector<vec> fbkv; // key
    vector<vec> fbvv; // value
    vector<vec> fbov;
    vector<vec> fbfv;
    vec fbsv;

    vec computeResult( vector<vec> attnInput, vec corefVector, uint wordIndex, bool verbose ) const;

  public:

    FModel( ) { }

    FModel( istream& is ) {
      while ( is.peek()=='F' ) {
        Delimited<char> c;
        is >> "F " >> c >> " ";
        if (c == 'P') is >> fwp >> "\n";
        if (c == 'p') is >> fbp >> "\n"; 
        if (c == 'I') {
          Delimited<int> i;
          DelimitedVector<psX, double, psComma, psX> vtemp;  
          is >> i >> " ";
          is >> vtemp >> "\n";
          assert (i == (int)fwi.size());
          fwi.push_back(vtemp);
        }
        if (c == 'i') {
          Delimited<int> i;
          DelimitedVector<psX, double, psComma, psX> vtemp;  
          is >> i >> " ";
          is >> vtemp >> "\n";
          assert (i == (int)fbi.size());
          fbi.push_back(vtemp);
        }
        if (c == 'O') {
          Delimited<int> i;
          DelimitedVector<psX, double, psComma, psX> vtemp;  
          is >> i >> " ";
          is >> vtemp >> "\n";
          assert (i == (int)fwo.size());
          fwo.push_back(vtemp);
        }
        if (c == 'o') {
          Delimited<int> i;
          DelimitedVector<psX, double, psComma, psX> vtemp;  
          is >> i >> " ";
          is >> vtemp >> "\n";
          assert (i == (int)fbo.size());
          fbo.push_back(vtemp);
        }
        if (c == 'F') {
          Delimited<int> i;
          DelimitedVector<psX, double, psComma, psX> vtemp;  
          is >> i >> " ";
          is >> vtemp >> "\n";
          assert (i == (int)fwf.size());
          fwf.push_back(vtemp);
        }
        if (c == 'f') {
          Delimited<int> i;
          DelimitedVector<psX, double, psComma, psX> vtemp;  
          is >> i >> " ";
          is >> vtemp >> "\n";
          assert (i == (int)fbf.size());
          fbf.push_back(vtemp);
        }
//        if (c == 'I') is >> fwi >> "\n";
//        if (c == 'i') is >> fbi >> "\n"; 
//        if (c == 'O') is >> fwo >> "\n";
//        if (c == 'o') is >> fbo >> "\n"; 
//        if (c == 'F') is >> fwf >> "\n";
//        if (c == 'f') is >> fbf >> "\n"; 
        if (c == 'S') is >> fws >> "\n";
        if (c == 's') is >> fbs >> "\n"; 
        if (c == 'H') {
          Delimited<int> h;
          is >> h >> "\n";
          num_heads = h;
        }
      }

      FSYN_DIM = FSYN_DIM_DEFAULT;
      while ( is.peek()=='C' ) {
        Delimited<char> c;
        Delimited<CVar> cv;
        DelimitedVector<psX, double, psComma, psX> vtemp;  
        is >> "C " >> c >> " " >> cv >> " ";
        is >> vtemp >> "\n";
        //if (c == 'B') mcbv.try_emplace(cv,vtemp);
        assert (c == 'B');
        mcbv.try_emplace(cv,vtemp);
        FSYN_DIM=vtemp.size();
      }

      FSEM_DIM = FSEM_DIM_DEFAULT;
      FANT_DIM = FANT_DIM_DEFAULT;
      //zeroCatEmb=arma::zeros(FSYN_SIZE);
      while ( is.peek()=='K' ) {
        Delimited<char> c;
        Delimited<K> k;
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "K " >> c >> " " >> k >> " ";
        is >> vtemp >> "\n";
        if (c == 'B') { 
          mkbdv.try_emplace(k, vtemp);
          FSEM_DIM=vtemp.size();
        }
        else if (c == 'F') { 
          mkfdv.try_emplace(k, vtemp);
          FSEM_DIM=vtemp.size();
        }
        else if (c == 'A') {
          mkadv.try_emplace(k, vtemp);
          FANT_DIM=vtemp.size();
        }
      }
      while ( is.peek()=='f' ) {
        unsigned int i;
        is >> "f " >> i >> " ";
        is >> mifek[i] >> "\n";
        mfeki[mifek[i]] = i;
      }

      fwpm = fwp;
      for ( vector<double> v : fwi ) {
        mat m = v;
        fwim.push_back(m);
      }
      for ( vector<double> v : fwo ) {
        mat m = v;
        fwom.push_back(m);
      }
      for ( vector<double> v : fwf ) {
        mat m = v;
        fwfm.push_back(m);
      }
//      fwim = fwi;
//      fwom = fwo;
//      fwfm = fwf;
      fwsm = fws;

      fbpv = fbp;
      for ( vector<double> v : fbi ) {
        vec vvec = v;
        fbiv.push_back(vvec);
      }
      for ( vector<double> v : fbo ) {
        vec vvec = v;
        fbov.push_back(vvec);
      }
      for ( vector<double> v : fbf ) {
        vec vvec = v;
        fbfv.push_back(vvec);
      }
//      fbiv = fbi;
//      fbov = fbo;
//      fbfv = fbf;
      fbsv = fbs;

      //cerr << "FSEM: " << FSEM_SIZE << " FSYN: " << FSYN_SIZE << " FANT: " << FANT_SIZE << endl;
      cerr << "FSEM: " << FSEM_DIM << " FSYN: " << FSYN_DIM << " FANT: " << FANT_DIM << endl;
      //FFULL_WIDTH = 8 + 2*FSEM_SIZE + FSYN_SIZE + FANT_SIZE;

      // reshape weight matrices
      uint pre_attn_dim = 7 + 2*FSEM_DIM + FSYN_DIM;
      attn_dim = fwp.size()/pre_attn_dim;
      // output of attn layer is concatenated with hvAnt (dim = FANT_DIM)
      // and nullA (dim = 1)
      //uint post_attn_dim = attn_dim + FANT_DIM + 1;
      uint hidden_dim = fwf[0].size()/attn_dim;
      // input to final feedforward is output of last fwfm concatenated
      // with coref vector and nullAnt bit
      uint fws_input_dim = hidden_dim + FANT_DIM + 1;
      uint output_dim = fws.size()/fws_input_dim;

      fwpm.reshape(attn_dim, pre_attn_dim);

      // fwim contains query, key, and value projection matrices,
      // each of dimension attn_dim x attn_dim
      for ( mat m : fwim ) {
        m.reshape(3*attn_dim, attn_dim);
        fwqm.push_back(m.rows(0, attn_dim-1));
        fwkm.push_back(m.rows(attn_dim, 2*attn_dim-1));
        fwvm.push_back(m.rows(2*attn_dim, 3*attn_dim-1));
      }
      for (uint i=0; i<fwom.size(); i++) {
        fwom[i].reshape(attn_dim, attn_dim);
      }

      for (uint i=0; i<fwfm.size(); i++) {
        fwfm[i].reshape(hidden_dim, attn_dim);
      }

      fwsm.reshape(output_dim, fws_input_dim);

      // fbiv contains biases vectors for query, key, and value
      for ( vec v : fbiv ) {
        fbqv.push_back(v(span(0, attn_dim-1)));
        fbkv.push_back(v(span(attn_dim, 2*attn_dim-1)));
        fbvv.push_back(v(span(2*attn_dim, 3*attn_dim-1)));
      }
      head_dim = attn_dim / num_heads;
    }

    const FEK& getFEK( unsigned int i ) const {
      auto it = mifek.find( i );
      assert( it != mifek.end() );
      if (it == mifek.end()) { 
        cerr << "ERROR: FEK not defined in fmodel: no value found for: " << i << endl; 
      }
      return it->second;
    }

    const vec getCatEmbed( CVar i, Delimited<char> c) const {
      vec zeroCatEmb = zeros(FSYN_DIM);
      assert (c == 'B');
      auto it = mcbv.find( i );
      return ( ( it != mcbv.end() ) ? it->second : zeroCatEmb );
    }

    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed;// = arma::zeros(FSEM_SIZE);
      if (c == 'B') {
        KVecEmbed = arma::zeros(FSEM_DIM);
        for ( auto& kV : hv.at(0) ) {
//          if ( kV == K::kTop) {
//            KVecEmbed += arma::ones(FSEM_DIM);
//            continue;
//          }
          if ( kV == K::kBot) {
            continue;
          }
          auto it = mkbdv.find( kV );
          if ( it == mkbdv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else if (c == 'F') {
        KVecEmbed = arma::zeros(FSEM_DIM);
        for ( auto& kV : hv.at(0) ) {
//          if ( kV == K::kTop) {
//            KVecEmbed += arma::ones(FSEM_DIM);
//            continue;
//          }
          if ( kV == K::kBot) {
            continue;
          }
          auto it = mkfdv.find( kV );
          if ( it == mkfdv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else if (c == 'A') {
        KVecEmbed = arma::zeros(FANT_DIM);
        for ( auto& kV : hv.at(0) ) {
//          if ( kV == K::kTop) {
//            KVecEmbed += arma::ones(FANT_DIM);
//            continue;
//          }
          if ( kV == K::kBot) {
            continue;
          }
          auto it = mkadv.find( kV );
          if ( it == mkadv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else cerr << "ERROR: F model KVec position misspecified." << endl;
      return KVecEmbed;
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) {
      const auto& it = mfeki.find( FEK(f,e,k) );  if( it != mfeki.end() ) return( it->second );
      mfeki[ FEK(f,e,k) ] = iNextResponse;  mifek[ iNextResponse ] = FEK(f,e,k);  return( iNextResponse++ );
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) const {                  // const version with closed predictor domain
      const auto& it = mfeki.find( FEK(f,e,k) );
      return ( ( it != mfeki.end() ) ? it->second : uint(-1) );
    }

    vec calcResponses( FPredictorVec& lfpredictors, int wordIndex, bool verbose ) const;

    void testCalcResponses() const;
};

vec FModel::calcResponses( FPredictorVec& lfpredictors, int wordIndex, bool verbose ) const {
// return distribution over FEK indices
  const HVec hvA = lfpredictors.getHvA();
  const bool nullA = lfpredictors.getNullA();
  const vec hvAEmb = getKVecEmbed(hvA, 'A');

  vec corefVec = join_cols(hvAEmb, zeros(1));
  if (nullA) corefVec(hvAEmb.size()) = 1;

  const BeamElement<HiddState> be = lfpredictors.getBeamElement();

  vector<vec> attnInput;
  
  //uint MAX_WINDOW_SIZE = 20;
  uint MAX_WINDOW_SIZE = 10;
  uint wordOffset = 0;
  // this moves backwards in time, starting with the word for which the
  // response is being calculated. wordOffset tracks how many
  // words back we've moved
  for (const BeamElement<HiddState>* curr = &be; ( (curr != &BeamElement<HiddState>::beStableDummy) && (wordOffset < MAX_WINDOW_SIZE) ); curr=&curr->getBack(), wordOffset++) {
    CVar catB = getCatBase(*curr);
    HVec hvB = getHvB(*curr);
    HVec hvF = getHvF(*curr);
    uint d = getDepth(*curr);
    
    vec catBEmb = getCatEmbed(catB, 'B');
    vec hvBEmb = getKVecEmbed(hvB, 'B');
    vec hvFEmb = getKVecEmbed(hvF, 'F');

    vec currAttnInput = join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), zeros(7)); 
    currAttnInput(2*FSEM_DIM + FSYN_DIM + d) = 1;
    // vector<> doesn't have an emplace_front method
    attnInput.emplace_back(currAttnInput);
  }

  // reverse attnInput so that the last item is the most recent word
  reverse(attnInput.begin(), attnInput.end());
  
  return computeResult(attnInput, corefVec, wordIndex, verbose);
}


// returns distribution over FEK indices
// attnInput contains the embeddings for previous words up to the current word
// attnInput.back() is the current word that we are making an F decision for
vec FModel::computeResult( vector<vec> attnInput, vec corefVec, uint wordIndex, bool verbose ) const {
  if ( verbose ) {
    for ( uint j=0; j<attnInput.size(); j++ ) { 
      cerr << "F word " << j << " pre-fwpm attn input" << endl;
      print_vec(attnInput[j]);
    }
//    cerr << "F linear bias" << endl;
//    print_vec(fbpv);
//    cerr << "F linear weight first col" << endl;
//    print_vec(fwpm.col(0));
  }
  bool usePositionalEncoding = false;
  mat currAttnInputs(fbpv.size(), attnInput.size());

  for ( uint i=0; i<attnInput.size(); i++ ) {
    vec proj = fwpm*attnInput[i] + fbpv;
    if ( verbose ) {
//      cerr << "F word " << i << " thing 1" << endl;
//      print_vec(fwpm.col(0));
//      cerr << "F word " << i << " thing 2" << endl;
//      //print_vec(attnInput[i]);
//      cerr << attnInput[i] << endl;
//      cerr << "F word " << i << " product, no bias" << endl;
//      print_vec(fwpm*attnInput[i]);
      cerr << "F word " << i << " proj" << endl;
      print_vec(proj);
    }
    if ( usePositionalEncoding ) {
      proj = proj + getPositionalEncoding(fbpv.size(), i);
    }
    currAttnInputs.col(i) = proj;
  }

  uint numTransformerLayers = fwqm.size();
  // used to scale the attention softmax (see section 3.2.1 of Attention
  // Is All You Need)
  const double scalingFactor = sqrt(head_dim);

  // alibi positional encodings, from this paper:
  // https://arxiv.org/pdf/2108.12409.pdf
  // TODO make it impossible to do both sinusoidal encodings and alibi
  vector<mat> alibiMats  = getAlibiMatrices( num_heads, attnInput.size() );
  for ( uint trLayer=0; trLayer<numTransformerLayers; trLayer++ ) {

    if ( verbose ) {
      cerr << " ==== transformer layer " << trLayer << " ====" << endl;
    }

    if ( verbose ) {
      for ( uint j=0; j<attnInput.size(); j++ ) { 
        cerr << "F word " << j << " curr attn inputs" << endl;
        print_vec(currAttnInputs.col(j));
      }
    }

    // each matrix contains the q/k/v for a single attn head
    mat queries[num_heads];
    mat keys[num_heads];
    mat values[num_heads];

    // initialize q/k/v matrices
    for ( uint i=0; i<num_heads; i++ ) {
      queries[i] = mat(attnInput.size(), head_dim);
      keys[i] = mat(attnInput.size(), head_dim);
      values[i] = mat(attnInput.size(), head_dim);
    }
    
    vec curr;
    for ( uint j=0; j<attnInput.size(); j++ ) { 
      vec curr = currAttnInputs.col(j);

      mat fwqm_i = fwqm[trLayer];
      vec fbqv_i = fbqv[trLayer];
      vec query = fwqm_i*curr+ fbqv_i;

      mat fwkm_i = fwkm[trLayer];
      vec fbkv_i = fbkv[trLayer];
      vec key = fwkm_i*curr+ fbkv_i;

      mat fwvm_i = fwvm[trLayer];
      vec fbvv_i = fbvv[trLayer];
      vec value = fwvm_i*curr+ fbvv_i;
      // iterate over attention heads
      for ( uint i=0; i < num_heads; i++ ) {
        mat query_i = query(span( i*head_dim, (i+1)*head_dim-1 ));
        queries[i].row(j) = query_i.t();

        mat key_i = key(span( i*head_dim, (i+1)*head_dim-1 ));
        keys[i].row(j) = key_i.t();

        mat value_i = value(span( i*head_dim, (i+1)*head_dim-1 ));
        values[i].row(j) = value_i.t();
      }
    }

    mat attnResult(attnInput.size(), attn_dim);
    // compute attn output head by head
    for ( uint i=0; i<num_heads; i++ ) {
      // sdp = scaled dot product
      mat sdp = (queries[i] * keys[i].t()) / scalingFactor;
      // add in linear bias for positional encoding
      sdp = sdp + alibiMats[i];
      if ( verbose ) {
        cerr << "F attn head " << i << endl;
        cerr << "F pre-softmax attn output weights:" << endl;
        for ( uint k=0; k<attnInput.size(); k++ ) {
          cerr << "F word " << k << endl;
          print_vec(sdp.row(k).t());
        }
      }
      sdp = exp(sdp);
      // take softmax of each row
      for ( uint j=0; j<attnInput.size(); j++ ) { 
        vec sdp_j = sdp.row(j).t();
        // mask any words later in time than the current word
        for ( uint k=j+1; k<attnInput.size(); k++ ) {
          sdp_j(k) = 0;
        }
        double norm = accu(sdp_j);
        sdp.row(j) = (sdp_j/norm).t();
      }
      // (seqLegth x seqLength) * (seqLength x headDim)  => (seqLength x headDim)
      mat perHeadAttnResult = sdp * values[i];
      attnResult.cols( i*head_dim, (i+1)*head_dim-1 ) = perHeadAttnResult;
    }

    if ( verbose ) {
      for ( uint j=0; j<attnInput.size(); j++ ) { 
        cerr << "F word " << j << " attn result" << endl;
        print_vec(attnResult.row(j).t());
      }
    }

    mat fwom_i = fwom[trLayer];
    vec fbov_i = fbov[trLayer];
    mat attnOutput = fwom_i*attnResult.t();
    attnOutput = attnOutput.each_col() + fbov_i;

    if ( verbose ) {
      for ( uint j=0; j<attnInput.size(); j++ ) { 
        cerr << "F word " << j << " attn output" << endl;
        print_vec(attnOutput.col(j));
      }
    }

    mat fwfm_i = fwfm[trLayer];
    vec fbfv_i = fbfv[trLayer];
    mat m = fwfm_i*attnOutput;
    m = m.each_col() + fbfv_i;
    if ( verbose ) {
      for ( uint j=0; j<attnInput.size(); j++ ) { 
        cerr << "F word " << j << " pre-relu feedforward output" << endl;
        print_vec(m.col(j));
      }
    }
    currAttnInputs = relu(m);
    if ( verbose ) {
      for ( uint j=0; j<attnInput.size(); j++ ) { 
        cerr << "F word " << j << " feedforward output" << endl;
        print_vec(currAttnInputs.col(j));
      }
    }

  } // end for trLayer

  // we only want the result from the last word
//  vec hiddenInput = join_cols(attnOutput, corefVec);
  vec finalWordFfOutput = currAttnInputs.col(attnInput.size() - 1);
  vec secondFfInput = join_cols(finalWordFfOutput, corefVec);
  if ( verbose ) {
    cerr << "F second ff input:" << endl;
    //print_vec(secondFfInput);
    cerr << secondFfInput << endl;
  }

  vec logScores = fwsm * secondFfInput + fbsv;
  if ( verbose ) {
    cerr << "F second ff output:" << endl;
    print_vec(logScores);
    //cerr << logScores << endl;
  }
  vec scores = exp(logScores);
  double outputNorm = accu(scores);
  vec result = scores/outputNorm;
  if ( verbose ) {
    cerr << "F result\n" << result << endl;
  }
  return result;
//  bool usePositionalEncoding = false;
//  vec last = attnInput.back();
//  if ( verbose ) {
//    //cerr << "F last" << last << endl;
//
////    cerr << "F fwpm" << endl;
////    cerr << "num rows: " << fwpm.n_rows << endl;
////    cerr << "num cols: " << fwpm.n_cols << endl;
////    for ( uint j=0; j<fwpm.n_cols; j++ ) {
////      for ( uint i=0; i<fwpm.n_rows; i++ ) {
////        cerr << fwpm(i, j) << endl;
////      }
////    }
////    cerr << "F fbpv" << fbpv << endl;
//  }
//  vec proj = fwpm*last + fbpv;
//  if ( usePositionalEncoding ) {
//    proj = proj + getPositionalEncoding(fbpv.size(), wordIndex);
//  }
//  if ( verbose ) {
//    cerr << "F proj" << proj << endl;
//  }
//  const vec query = fwqm*proj + fbqv;
//  // used to scale the attention softmax (see section 3.2.1 of Attention
//  // Is All You Need)
//  const double scalingFactor = sqrt(fbqv.size());
//
//  vector<vec> values;
//  vector<double> scaledDotProds;
//  int currIndex = wordIndex - attnInput.size() + 1;
//  for ( vec curr : attnInput ) { 
//    proj = fwpm*curr + fbpv;
//    if ( usePositionalEncoding ) {
//      proj = proj + getPositionalEncoding(fbpv.size(), currIndex++);
//    }
//    vec key = fwkm*proj + fbkv;
//    vec value = fwvm*proj + fbvv;
//    values.emplace_back(value);
//    scaledDotProds.emplace_back(dot(query, key)/scalingFactor);
//  }
//
//  vec sdp = vec(scaledDotProds.size());
//  for ( uint i=0; (i < scaledDotProds.size()); i++ ) {
//    sdp(i) = scaledDotProds[i];
//  }
//
//  vec sdpExp = exp(sdp);
//  double norm = accu(sdpExp);
//  vec sdpSoftmax = sdpExp/norm;
//
//  // calculate scaled_softmax(QK)*V
//  vec attnResult = zeros<vec>(fbvv.size());
//
//  for ( uint i=0; (i < values.size()); i++ ) {
//    double weight = sdpSoftmax(i);
//    vec val = values[i];
//    attnResult += weight*val;
//  }
//
//  vec attnOutput = fwom*attnResult + fbov;
//
//  if ( verbose ) {
//    cerr << "F attnOutput" << attnOutput << endl;
//  }
//  // final bit is for nullA
//  vec hiddenInput = join_cols(attnOutput, corefVec);
//  vec logScores = fwsm * relu(fwfm*hiddenInput + fbfv) + fbsv;
//  vec scores = exp(logScores);
//  double outputNorm = accu(scores);
//  vec result = scores/outputNorm;
//  if ( verbose ) {
//    cerr << "F result" << result << endl;
//  }
//  return result;
} 


// pass vectors of 1s as inputs as a sanity check (for comparison with
// Python training code)
void FModel::testCalcResponses() const {
  //vec corefVec = ones<vec>(FANT_DIM+1);
  vec corefVec;
  vec currAttnInput;
  vector<vec> attnInput;
  uint attnInputDim = 2*FSEM_DIM + FSYN_DIM + 7;
  uint corefDim = FANT_DIM + 1;
  uint seqLength = 5;
  // the ith dummy input will be a vector of 0.1*(i+1)
  // [0.1 0.1 0.1 ...] [0.2 0.2 0.2 ...] ...
  for ( uint i=0; (i<seqLength); i++ ) {
    cerr << "F ==== output for word " << i << " ====" << endl;
    currAttnInput = vec(attnInputDim);
    currAttnInput.fill(0.1 * (i+1));
    //attnInput.emplace_back(ones<vec>(2*FSEM_DIM + FSYN_DIM + 7));
    attnInput.emplace_back(currAttnInput);
    corefVec = vec(corefDim);
    corefVec.fill(0.1 * (i+1));
    computeResult(attnInput, corefVec, i, 1);
  }
}


  //=============================================================================

//  CVar catB = getCatBase(be);
//  HVec hvB = getHvB(be);
//  HVec hvF = getHvF(be);
//  uint d = getDepth(be);
//  
//  vec catBEmb = getCatEmbed(catB, 'B');
//  vec hvBEmb = getKVecEmbed(hvB, 'B');
//  vec hvFEmb = getKVecEmbed(hvF, 'F');
//
//  // populate input vector to pre-attn feedforward
//  vec preAttnInputs = join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), zeros(7)); 
//  preAttnInputs(2*FSEM_DIM + FSYN_DIM + d) = 1;
//
//  vec attnInputs = fwpm*preAttnInputs + fbpv;
//  // we only char about the query for the latest word, hence the const
//  const vec query = fwqm*attnInputs + fbqv;
//  vec key = fwkm*attnInputs + fbkv;
//  vec value = fwvm*attnInputs + fbvv;
//
//  // TODO not sure if this is the right way to get the length of a vec
//  const double scalingFactor = sqrt(fbqv.size());
//
//  list<vec> values;
//  // Q*K for each f decision being attended to, scaled by sqrt(attn_dim)
//  list<double> scaledDotProds;
//  values.emplace_front(value);
//  scaledDotProds.emplace_front(dot(query, key)/scalingFactor);
//
//  //const BeamElement<HiddState>* pbeAnt = &beDummy;
//  //const BeamElement<HiddState>* curr = &be.getBack();
//
//  //while (&curr != &BeamElement<HiddState>::beStableDummy)
//  //for ( int tAnt = t; (&pbeAnt->getBack() != &BeamElement<HiddState>::beStableDummy) && (int(t-tAnt)<=COREF_WINDOW); tAnt--, pbeAnt = &pbeAnt->getBack()) { 
//  
//  int wordOffset = 0;
//  // TODO limit how far back you go?
//  for (const BeamElement<HiddState>* curr = &be.getBack(); (curr != &BeamElement<HiddState>::beStableDummy); curr = &curr->getBack(), wordOffset++) {
//    catB = getCatBase(*curr);
//    hvB = getHvB(*curr);
//    hvF = getHvF(*curr);
//    d = getDepth(*curr);
//    
//    catBEmb = getCatEmbed(catB, 'B');
//    hvBEmb = getKVecEmbed(hvB, 'B');
//    hvFEmb = getKVecEmbed(hvF, 'F');
//
//    preAttnInputs = join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), zeros(7)); 
//    preAttnInputs(2*FSEM_DIM + FSYN_DIM + d) = 1;
//
//    attnInputs = fwpm*preAttnInputs + fbpv;
//    // TODO add positional encoding here, using wordIndex - wordOffset
//    key = fwkm*attnInputs + fbkv;
//    value = fwvm*attnInputs + fbvv;
//    values.emplace_front(value);
//    scaledDotProds.emplace_front(dot(query, key)/scalingFactor);
//
//    //curr = &curr->getBack();
//  }
//
//  // take softmax of scaled dot products
//  vec sdp = vec(scaledDotProds.size());
//  for ( uint i=0; (i < scaledDotProds.size()); i++ ) {
//    sdp(i) = scaledDotProds.front();
//    scaledDotProds.pop_front();
//  }
//  vec sdpExp = exp(sdp);
//  double norm = accu(sdpExp);
//  vec sdpSoftmax = sdpExp/norm;
//
//  // calculate scaled_softmax(QK)*V
//  vec attnResult = zeros<vec>(fbvv.size());
//
//  for ( uint i=0; (i < values.size()); i++ ) {
//    double weight = sdpSoftmax(i);
//    vec val = values.front();
//    values.pop_front();
//    attnResult = attnResult + weight*val;
//  }
//
//  vec attnOutput = fwom*attnResult + fbov;
//  // final bit is for nullA
//  //vec hiddenInput = join_cols(join_cols(attnResult, hvAEmb), zeros(1));
//  vec hiddenInput = join_cols(join_cols(attnOutput, hvAEmb), zeros(1));
//  //if (nullA) hiddenInput(attnResult.size() + hvAEmb.size()) = 1;
//  if (nullA) hiddenInput(attnOutput.size() + hvAEmb.size()) = 1;
//  vec logScores = fwsm * relu(fwfm*hiddenInput + fbfv) + fbsv;
//  vec scores = exp(logScores);
//  double outputNorm = accu(scores);
//  return scores/outputNorm;
//} 
