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

////////////////////////////////////////////////////////////////////////////////

class NPredictorVec {
  private:
     int mdist;
     const HVec& basesem;
     const HVec& antecedentsem;
     CVar        basec;
     CVar        antecedentc;
     bool corefON;

  public:
    //constructor
    template<class LM>
    NPredictorVec( LM& lm, const Sign& candidate, bool bcorefON, int antdist, const StoreState& ss, bool ABLATE_UNARY ) :
      basesem (( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot ), 
      antecedentsem ((candidate.getHVec().size() > 0 ) ? candidate.getHVec() : hvBot ) { 
      mdist = antdist;
      antecedentc = candidate.getCat();
      basec = ss.getBase().getCat();
      corefON = bcorefON;
      //ABLATE_UNARY not defined for dense semproc since NN implicitly captures/can capture joint feature relations
    }

    //accessors
    int   getAntDist()   const { return mdist;   }
    int   getAntDistSq() const { return mdist * mdist; }
    const HVec& getBaseSem()   const { return basesem; }
    const HVec& getAnteSem()   const { return antecedentsem; }
    CVar  getBaseC()     const { return basec; }
    CVar  getAnteC()     const { return antecedentc; }
    bool  getCorefOn()   const { return corefON; }

    friend ostream& operator<< ( ostream& os, const NPredictorVec& mv ) {
      os << " " << mv.getBaseC() << " " << mv.getAnteC() << " " << mv.getBaseSem() << " " << mv.getAnteSem() << " " << mv.getAntDist() << " " << mv.getAntDistSq() << " " << mv.getCorefOn();
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////

class NModel {
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> KDenseVec;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;

  uint NSYN_SIZE = 10; //these will all be overwritten
  uint NSEM_SIZE = 20;
  uint NFULL_WIDTH = 13; 

  private:
    mat nwm;
    mat nwsm;
    vec nwbv;
    vec nwsbv;
    DelimitedVector<psX, double, psComma, psX> nwb; // n model weights (first layer), biases
    DelimitedVector<psX, double, psComma, psX> nwsb; // n model weights (second layer), biases
    DelimitedVector<psX, double, psComma, psX> nw;                              // n model weights. square for now, doesn't have to be
    DelimitedVector<psX, double, psComma, psX> nws; // n model weights (second layer), vector for reading in, to be unsqueezed before use
    map<CVar,CVec> mcv; //map between cat and 10d embeddings
    CVec zeroCatEmb;
    map<KVec,KDenseVec> mkdv;                  // map between KVec and embeds

  public:
    NModel( ) : zeroCatEmb(NSYN_SIZE) { }
    NModel( istream& is ) : zeroCatEmb(NSYN_SIZE) {
      //DelimitedMat<psX, double, psComma, NPREDDIM, NPREDDIM, psX> nw;                              // n model weights. square for now, doesn't have to be

      //rewrite how to read in N model learned weights, crib off of J model
      while( is.peek()=='N' ) {
        Delimited<char> c;
        is >> "N " >> c >> " ";
        if (c == 'F') is >> nw >> "\n";
        if (c == 'f') is >> nwb >> "\n";
        if (c == 'S') is >> nws >> "\n";
        if (c == 's') is >> nwsb >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        DelimitedVector<psX, double, psComma, psX> vtemp;  
        is >> "C " >> c >> " ";
        is >> vtemp >> "\n";
        mcv.try_emplace(c,vtemp);
        NSYN_SIZE=vtemp.size();
      }
      zeroCatEmb=CVec(NSYN_SIZE);
      while ( is.peek()=='K' ) {
        Delimited<K> k;
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "K " >> k >> " ";
        is >> vtemp >> "\n";
        mkdv.try_emplace(k,vtemp);
        //is >> mkdv.try_emplace(k,NSEM_SIZE).first->second >> "\n";
        NSEM_SIZE=vtemp.size();
      }
      NFULL_WIDTH = 2*NSEM_SIZE+2*NSYN_SIZE+3;
      //Reshape read-in model params to correct dimensions, save to private members
      //first layer
      nwm = nw;
      nwm.reshape(nwm.size()/NFULL_WIDTH, NFULL_WIDTH); //hidden x input. converts input dim to hidden dim
      //cerr << "nwm reshaped n_rows, ncols, size: " << nwm.n_rows << " " << nwm.n_cols << " " << nwm.size() << endl;
      nwbv = nwb;
      //cerr << "nwbv size: " << nwbv.size() << endl;

      //second layer
      nwsm = nws;
      nwsm.reshape(nwsm.size()/(nwm.size()/NFULL_WIDTH), nwm.size()/NFULL_WIDTH); //unsqueeze vector to matrix.  //outputdim, hidden. converts hidden dim to outputdim
      //cerr << "nwsm reshaped n_rows, n_cols, size: " << nwsm.n_rows << " " << nwsm.n_cols << " " << nwsm.size() << endl;
      nwsbv = nwsb;
      //cerr << "nwsbv size: " << nwsbv.size() << endl;
      
      //print out first row
      //cerr << "first nwm matrix col: " << endl;
      //for (uint i = 0; i < nwm.n_rows; i++) { cerr << nwm(i,0) << ","; }
      //cerr << endl;
    }

    const CVec& getCatEmbed( CVar i ) const {
      //cerr << "attempting to find cat with index i: " << i << endl;
      //cerr << "mcv size: " << mcv.size() << endl;
      auto it = mcv.find( i );
      if (it == mcv.end()) { 
        cerr << "ERROR: CVar not defined in nmodel: no embedding found for: " << i << endl; 
        return zeroCatEmb;
      }
      assert( it != mcv.end() );
      return it->second;
      //}
    }
  
    const KDenseVec getKVecEmbed( HVec hv ) const {
      KDenseVec KVecEmbed = KDenseVec(arma::zeros(NSEM_SIZE));
      for ( auto& kV : hv.at(0) ) {
        if ( kV == K::kTop) {
          KVecEmbed += KDenseVec(arma::ones(NSEM_SIZE));
          continue;
        }
        auto it = mkdv.find( kV );
        if ( it == mkdv.end() ) {
          continue;
        } else {
          KVecEmbed += it->second;
        }
      }
      return KVecEmbed;
    }

    arma::vec calcResponses( const NPredictorVec& npv ) const {
      //cerr << "entering calcLogResponses..." << endl;
      arma::vec nlogresponses = arma::zeros( NFULL_WIDTH); 
      
      //cerr << "getting cat indices, hvecs, ad-hoc feat values..." << endl;
      CVar catB = npv.getBaseC();
      CVar catA = npv.getAnteC();
      const HVec& hvB = npv.getBaseSem();
      const HVec& hvA = npv.getAnteSem();
      int antdist = npv.getAntDist();
      int antdistsq = npv.getAntDistSq();
      bool corefon = npv.getCorefOn();

      //cerr << "getting cat embeds..." << endl;
      const CVec& catBEmb = getCatEmbed(catB);
      const CVec& catAEmb = getCatEmbed(catA);
      const KDenseVec& hvBEmb = getKVecEmbed(hvB);
      const KDenseVec& hvAEmb = getKVecEmbed(hvA);

      //populate predictor vec by catting vecs or else filling in nlogresponses' values
      //cerr << "populating predictor vec with feats..." << endl;
      //arma::Col<double> feats = std::vector<double> {antdist, antdistsq, corefon}; 
      for(unsigned int i = 0; i < catBEmb.n_elem; i++){ nlogresponses(i) = catBEmb(i); }
      for(unsigned int i = 0; i < catAEmb.n_elem; i++){ nlogresponses(catBEmb.n_elem+i) = catAEmb(i); }
      //cerr << "inserting hvec feats into predictor vec ..." << endl;
      for(unsigned int i = 0; i < hvBEmb.n_elem; i++){ nlogresponses(catBEmb.n_elem+catAEmb.n_elem+i) = hvBEmb(i); }
      for(unsigned int i = 0; i < hvAEmb.n_elem; i++){ nlogresponses(catBEmb.n_elem+catAEmb.n_elem+hvBEmb.n_elem+i) = hvAEmb(i); }
      int denseendind = catBEmb.n_elem+catAEmb.n_elem+hvBEmb.n_elem+hvAEmb.n_elem; 
      //cerr << "inserting ad-hoc feats into predictor vec ..." << endl;
      nlogresponses(denseendind+0) = antdist;
      nlogresponses(denseendind+1) = antdistsq;
      nlogresponses(denseendind+2) = corefon;
      //arma::vec nlogresponses = arma::join_cols(catBEmb, catAEmb, hvBEmb, hvAEmb); //, feats);
      ////later potentially add gender animacy etc features, if continue to have inconsistency issues with number, gender, etc.
      //cerr << "catB: " << catB << " catA: " << catA << " hvB: " << hvB << " hvA: " << hvA << " antdist: " << antdist << " sqantdist: " << antdistsq << " corefon: " << corefon << endl;
      //cerr << "catBEmb: catBEmb << catAEmb << hvBEmb << hvAEmb << endl;
      //cerr << nlogresponses << endl;
      //cerr << "feature multiplication by weights..." << endl;
      //push through weight matrix, norm and return score
      //cout << "NPREDDIM: " << NPREDDIM << endl;
      //cerr << "nlogresponses size: " << nlogresponses.size() << endl;
      //cerr << "nwm row,cols: " << nwm.n_rows << "," << nwm.n_cols << endl;
      //cerr << "nwsm row,cols: " << nwsm.n_rows << "," << nwsm.n_cols << endl;
      arma::vec nlogscores = nwsm * relu(nwm*nlogresponses+nwbv) + nwsbv;
      arma::vec nscores = arma::exp(nlogscores);
      double nnorm = arma::accu(nscores);

      return nscores/nnorm;
    }
};

