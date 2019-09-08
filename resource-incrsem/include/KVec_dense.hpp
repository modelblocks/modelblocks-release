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

//class KVec : public DelimitedCol<psLBrack, double, psComma, 20, psRBrack> {
class KVec : public DelimitedCol<psLBrack, double, psComma, 40, psRBrack> {
  public:
    KVec ( ) { }
//    KVec ( const Col<double>& kv ) : DelimitedCol<psLBrack, double, psComma, 20, psRBrack>(kv) { }
    KVec ( const Col<double>& kv ) : DelimitedCol<psLBrack, double, psComma, 40, psRBrack>(kv) { }
    KVec& add( const KVec& kv ) { *this += kv; return *this; }
};
//const KVec kvTop   ( arma::ones<Col<double>>(20)  );
//const KVec kvBot   ( arma::zeros<Col<double>>(20) );
//const KVec kvDitto ( arma::randn<Col<double>>(20) );
const KVec kvTop   ( arma::ones<Col<double>>(40)  );
const KVec kvBot   ( arma::zeros<Col<double>>(40) );
const KVec kvDitto ( arma::randn<Col<double>>(40) );

////////////////////////////////////////////////////////////////////////////////

class EMat {
  map<XVar,KVec> mxv;
  public:
    EMat() {}
    EMat(istream& is) {
      while ( is.peek()=='E' ) {
        Delimited<XVar> x;
        is >> "E " >> x >> " ";
        is >> mxv[x] >> "\n";
      }
    }
    KVec operator() ( XVar x ) const { const auto& it = mxv.find( x ); return ( it == mxv.end() ) ? KVec() : it->second; }   // return mxv[x]; }
    friend ostream& operator<< ( ostream& os, const EMat& matE ) {
      for ( const auto& it : matE.mxv ) os << it.first << " : " << it.second << endl;
      return os;
    }
// should return the vectors that underwent the -0 relationship function
};

////////////////////////////////////////////////////////////////////////////////

arma::mat relu( const arma::mat& km ) {
  arma::mat A(km.n_rows, 1);
  for ( unsigned int c = 0; c<km.n_rows; c++ ) {
    if ( km(c,0) <= 0 ) {A(c,0)=(0.0);}
    else A(c,0) = (km(c));
  }
  return A;
}

//class OFunc {
////  map<int,DelimitedMat<psX, double, psComma, 40, 20, psX>> mrwf;
////  map<int,DelimitedMat<psX, double, psComma, 20, 40, psX>> mrws;
//  map<int,DelimitedMat<psX, double, psComma, 80, 40, psX>> mrwf;
//  map<int,DelimitedMat<psX, double, psComma, 40, 80, psX>> mrws;
//  public:
//    OFunc() {}
//    OFunc(istream& is) {
//      while ( is.peek()=='O' ) {
//        Delimited<int> k;
//        Delimited<char> c;
//        is >> "O " >> k >> " " >> c >> " ";
//        if (c == 'F') is >> mrwf[k] >> "\n";
//        if (c == 'S') is >> mrws[k] >> "\n";
//      }
//    }
//
////  implementation of MLP; apply appropriate weights via matmul
//    arma::vec operator() ( int rel, const Col<double>& kv ) const {
////                          (20x40) * (40x20) * (20x1)
//      auto its = mrws.find(rel);
//      auto itf = mrwf.find(rel);
//      assert (its != mrws.end() && itf != mrwf.end());
//      return Mat<double>(its->second) * relu(Mat<double>(itf->second)*kv);
//    }
//};

class OFunc {
  map<int,DelimitedMat<psX, double, psComma, 40, 40, psX>> mrwf;
  public:
    OFunc() {}
    OFunc(istream& is) {
      while ( is.peek()=='O' ) {
        Delimited<int> k;
        is >> "O " >> k >> " ";
        is >> mrwf[k] >> "\n";
      }
    }

//  implementation of MLP; apply appropriate weights via matmul
    arma::vec operator() ( int rel, const Col<double>& kv ) const {
//                          (20x40) * (40x20) * (20x1)
      auto itf = mrwf.find(rel);
      assert (itf != mrwf.end());
//      return Mat<double>(itf->second) % kv;
      return Mat<double>(itf->second) * kv;
    }
};