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

#include <iostream>
#include <fstream>
#include <list>
#include <algorithm>
#include <thread>
#include <mutex>
#include <random>
using namespace std;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>

////////////////////////////////////////////////////////////////////////////////

char psOpenParen[]  = "(";
char psCloseParen[] = ")";

typedef Delimited<string> L;
typedef Delimited<int> Z;

class ForestTree : public DelimitedList<psX,ForestTree,psSpace,psX> {
 private:
  L l;
  arma::vec v;
  Z z;
 public:
  ForestTree ( )                : DelimitedList<psX,ForestTree,psSpace,psX> ( )        { }
  ForestTree ( const char* ps ) : DelimitedList<psX,ForestTree,psSpace,psX> ( ), l(ps) { }
  friend pair<istream&,ForestTree&> operator>> ( istream& is, ForestTree& t ) {
    return pair<istream&,ForestTree&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,ForestTree&> ist, const char* psDelim ) {
    if ( ist.first.peek()=='(' ) return ist.first >> "(" >> ist.second.l >> " " >> (DelimitedList<psX,ForestTree,psSpace,psX>&)ist.second >> ")" >> psDelim;
    else                         return ist.first >> ist.second.l >> psDelim;
  }
  friend bool operator>> ( pair<istream&,ForestTree&> ist, const vector<const char*>& vpsDelim ) {
    if ( ist.first.peek()=='(' ) return ist.first >> "(" >> ist.second.l >> " " >> (DelimitedList<psX,ForestTree,psSpace,psX>&)ist.second >> ")" >> vpsDelim;
    else                         return ist.first >> ist.second.l >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const ForestTree& t ) {
    if ( t.size()>0 ) return os << psOpenParen << t.l << psSpace << (DelimitedList<psX,ForestTree,psSpace,psX>&)t << psCloseParen;
    else              return os << t.l;
  }
  operator const L ( ) const { return l; }

  void addCounts ( map<L,arma::vec>& cPi, map<trip<L,L,L>,arma::mat>& cTheta, map<pair<L,L>,arma::vec>& cXi, uint numT, bool bRoot = true ) {
    if( bRoot )                               { arma::vec& cv = cPi[l];                                    if(cv.size()==0) cv = arma::vec(numT);           else cv( z )++; }
    if     ( size()==1 && front().size()==0 ) { arma::vec& cv = cXi[pair<L,L>(l,front().l)];               if(cv.size()==0) cv = arma::vec(numT);           else cv( z )++; }
    else if( size()==2                      ) { arma::mat& cm = cTheta[trip<L,L,L>(l,front().l,back().l)]; if(cm.size()==0) cm = arma::mat(numT,numT*numT); else cm( z, front().z*numT + back().z )++;
                                                for( auto& subtree : *this ) subtree.addCounts( cPi, cTheta, cXi, numT, false ); }
    else cerr << "ERROR -- tree not CNF: " << *this << endl;
  }

  void calcForest ( const map<trip<L,L,L>,arma::mat>& mTheta, const map<pair<L,L>,arma::vec>& mXi ) {
    if     ( size()==1 && front().size()==0 ) { v = mXi.find( pair<L,L>(l,front().l) )->second; }
    else if( size()==2                      ) { for( auto& subtree : *this ) subtree.calcForest( mTheta, mXi );
                                                v = mTheta.find( trip<L,L,L>(l,front().l,back().l) )->second * arma::vectorise( back().v * front().v.t() ); }
    else cerr << "ERROR -- tree not CNF: " << *this << endl;
  }

  void sampleZ ( const map<L,arma::vec>& mPi, const map<trip<L,L,L>,arma::mat>& mTheta, mt19937& e, bool bRoot = true ) {
    if( bRoot )     { arma::vec post = mPi.find( l )->second % v; discrete_distribution<uint> d( post.begin(), post.end() ); z = d(e); }
    if( size()==2 ) { arma::vec post = mTheta.find( trip<L,L,L>(l,front().l,back().l) )->second.row(z).t() % arma::vectorise( back().v * front().v.t() );
                      discrete_distribution<uint> d( post.begin(), post.end() );
                      uint zz = d(e); front().z = zz/v.size(); back().z = zz%v.size();
                      for( auto& subtree : *this ) subtree.sampleZ( mPi, mTheta, e, false ); }
  }

  double getProb ( const map<L,arma::vec>& mPi, const map<trip<L,L,L>,arma::mat>& mTheta, map<pair<L,L>,arma::vec>& mXi, bool bRoot = true ) {
    double pr = 1.0;
    if( bRoot )                               pr = mPi.find( l )->second(z);
    if(      size()==1 && front().size()==0 ) return pr * mXi.find( pair<L,L>(l,front().l) )->second(z);
    else if( size()==2 )                      { pr *= mTheta.find( trip<L,L,L>(l,front().l,back().l) )->second( z, v.size()*front().z + back().z );
                                                for( auto& subtree : *this ) pr *= subtree.getProb( mPi, mTheta, mXi, false ); return pr; }
    cerr << "ERROR -- tree not CNF: " << *this << endl;
    return 0.0;
  }
};


////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  uint numTypes   = 10;
  uint numIters   = 100;
  uint numThreads = 10;

  // Read trees...
  list<ForestTree> ltCorpus;
  while ( cin && EOF!=cin.peek() ) {
    ltCorpus.emplace_back(); ForestTree& t = ltCorpus.back();
    cin >> t >> "\n";
//    cout << "TREE: " << t << "\n";
  }

  double pseudocount = 0.2;
  // Init root, nonterm, term counts...
  map<L,          arma::vec> cPi;
  map<trip<L,L,L>,arma::mat> cTheta;
  map<pair<L,L>,  arma::vec> cXi;
  // Init root, nonterm, term models...
  map<L,          arma::vec> mPi;
  map<trip<L,L,L>,arma::mat> mTheta;
  map<pair<L,L>,  arma::vec> mXi;

  mt19937 e( random_device{}() );

  // For each iteration...
  for( uint i=0; i<numIters; i++ ) {

    ////// I. Sample models given counts from trees...

    // Obtain counts...
    for( auto& t : ltCorpus ) t.addCounts( cPi, cTheta, cXi, numTypes );

    // Update pi models...
    for( auto& lv : cPi ) {
      if( mPi[lv.first].size()==0 ) mPi[lv.first] = arma::vec( numTypes );
      for( uint j=0; j<numTypes; j++ ) { gamma_distribution<double> d( pseudocount+lv.second(j), 1.0 ); mPi[lv.first](j) = d(e); }
      mPi[lv.first] = arma::normalise( mPi[lv.first], 1 );
      lv.second.zeros();
    }
    if( i==0 ) for( auto& lllm : cTheta ) mTheta[lllm.first] = arma::mat( numTypes, numTypes*numTypes );
    // Update theta models...
    vector<thread> vtWorkers;
    for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {
      uint iCtr=0; for( auto& lllm : cTheta ) if( iCtr++%numThreads == numt ) {
//        if( mTheta[lllm.first].size()==0 ) mTheta[lllm.first] = arma::mat( numTypes, numTypes*numTypes );
        for( uint j=0; j<numTypes; j++ ) for( uint k=0; k<numTypes*numTypes; k++ ) { gamma_distribution<double> d( pseudocount+lllm.second(j,k), 1.0 ); mTheta[lllm.first](j,k) = d(e); }
        mTheta[lllm.first] = arma::normalise( mTheta[lllm.first], 1, 1 );
        lllm.second.zeros();
      }
    }, numtglobal ));
    for( auto& w : vtWorkers ) w.join();
    // Update xi m(dels...
    for( auto& lv : cXi ) {
      if( mXi[lv.first].size()==0 ) mXi[lv.first] = arma::vec( numTypes );
      for( uint j=0; j<numTypes; j++ ) { gamma_distribution<double> d( pseudocount+lv.second(j), 1.0 ); mXi[lv.first](j) = d(e); }
      mXi[lv.first] = arma::normalise( mXi[lv.first], 1 );
      lv.second.zeros();
    }

    ////// II. Sample trees given models...

    // For each thread...
    vector<thread> vtWorkers2;
    for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers2.push_back( thread( [&] (uint numt) {
      mt19937 e( random_device{}() );
      // For each tree in process's modulo partition of corpus...
      uint itree=0; for( auto& t : ltCorpus ) if( itree++%numThreads == numt ) {
        // A. Calc forest from model, bottom-up...
        t.calcForest( mTheta, mXi );
        // B. Sample tree from forest, top-down...
        t.sampleZ( mPi, mTheta, e );
      }
    }, numtglobal ));

    for( auto& w : vtWorkers2 ) w.join();

    double lgpr = 0.0;
    for( auto& t : ltCorpus ) { lgpr += log( t.getProb(mPi,mTheta,mXi) ); }

    cerr<<"Iteration "<<i<<": logprob="<<lgpr<<endl;
  }

  for( auto& lv : mPi ) for( uint j=0; j<numTypes; j++ )
    cout << "R : " << lv.first     << "_" << j
         << " = "   << lv.second(j) << endl;
  for( auto& lllm : mTheta ) for( uint j=0; j<numTypes; j++ ) for( uint k=0; k<numTypes*numTypes; k++ )
    cout << "G " << lllm.first.first()  << "_" << j
         << " : " << lllm.first.second() << "_" << k/numTypes
         <<  " "  << lllm.first.third()  << "_" << k%numTypes
         << " = " << lllm.second(j,k)    << endl;
  for( auto& llv : mXi ) for( uint j=0; j<numTypes; j++ )
    cout << "X " << llv.first.first  << "_" << j
         << " : " << llv.first.second 
         << " = " << llv.second(j)    << endl;
}

