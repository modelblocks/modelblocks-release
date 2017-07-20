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

DiscreteDomain<int> domW;
class W : public Delimited<DiscreteDomainRV<int,domW>> {
 public:
  W ( )                : Delimited<DiscreteDomainRV<int,domW>> ( )    { }
  W ( int i )          : Delimited<DiscreteDomainRV<int,domW>> ( i )  { }
  W ( const char* ps ) : Delimited<DiscreteDomainRV<int,domW>> ( ps ) { }
};

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domL;
class L : public Delimited<DiscreteDomainRV<int,domL>> {
 public:
  L ( )                : Delimited<DiscreteDomainRV<int,domL>> ( )    { }
  L ( int i )          : Delimited<DiscreteDomainRV<int,domL>> ( i )  { }
  L ( const char* ps ) : Delimited<DiscreteDomainRV<int,domL>> ( ps ) { }
};

////////////////////////////////////////////////////////////////////////////////

char psOpenParen[]  = "(";
char psCloseParen[] = ")";

//typedef Delimited<string> L;
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
    else                         { istream& os = ist.first >> ist.second.l >> psDelim; W w(ist.second.l.getString().c_str()); return os; }
  }
  friend bool operator>> ( pair<istream&,ForestTree&> ist, const vector<const char*>& vpsDelim ) {
    if ( ist.first.peek()=='(' ) return ist.first >> "(" >> ist.second.l >> " " >> (DelimitedList<psX,ForestTree,psSpace,psX>&)ist.second >> ")" >> vpsDelim;
    else                         { bool b = ist.first >> ist.second.l >> vpsDelim; W w(ist.second.l.getString().c_str()); return b; }
  }
  friend ostream& operator<< ( ostream& os, const ForestTree& t ) {
    if ( t.size()>0 ) return os << psOpenParen << t.l << "_" << t.z << psSpace << (DelimitedList<psX,ForestTree,psSpace,psX>&)t << psCloseParen;
    else              return os << t.l;
  }
  operator const L ( ) const { return l; }

  void init ( map<L,arma::vec>& cPi, map<trip<L,L,L>,arma::mat>& cTheta, map<L,arma::mat>& cXi, uint numT, bool bRoot = true ) {
    if( bRoot )                               { arma::vec& cv = cPi[l];                                                    if( cv.size()==0 ) cv = arma::zeros(numT);                }
    if     ( size()==1 && front().size()==0 ) { arma::mat& cm = cXi[l];                                                    if( cm.size()==0 ) cm = arma::zeros(numT,domW.getSize()); }
    else if( size()==1 || size()==2         ) { arma::mat& cm = cTheta[trip<L,L,L>(l,front().l,(size()==1)?"-":back().l)]; if( cm.size()==0 ) cm = arma::zeros(numT,numT*numT);
                                                for( auto& subtree : *this ) subtree.init( cPi, cTheta, cXi, numT, false ); }
    else cerr << "ERROR -- n-nary tree: " << *this << endl;
  }

  void addCounts ( map<L,arma::vec>& cPi, map<trip<L,L,L>,arma::mat>& cTheta, map<L,arma::mat>& cXi, uint numT, bool bRoot = true ) {
    if( bRoot )                               { arma::vec& cv = cPi[l];                                                    if( v.size()>0 ) cv( z )++;                               }
    if     ( size()==1 && front().size()==0 ) { arma::mat& cm = cXi[l];                                                    if( v.size()>0 ) cm( z, W(front().l.getString().c_str()).toInt() )++; }
    else if( size()==1 || size()==2         ) { arma::mat& cm = cTheta[trip<L,L,L>(l,front().l,(size()==1)?"-":back().l)]; if( v.size()>0 ) cm( z, front().z*numT + back().z )++;
                                                for( auto& subtree : *this ) subtree.addCounts( cPi, cTheta, cXi, numT, false );                    }
    else cerr << "ERROR -- n-ary tree: " << *this << endl;
  }

  void calcForest ( const map<trip<L,L,L>,arma::mat>& mTheta, const map<L,arma::mat>& mXi ) {
    if     ( size()==1 && front().size()==0 ) { v = mXi.find( l )->second.col( W(front().l.getString().c_str()).toInt() ); }
    else if( size()==1 || size()==2         ) { for( auto& subtree : *this ) subtree.calcForest( mTheta, mXi );
                                                if( size()==1 ) v = mTheta.find( trip<L,L,L>(l,front().l,"-") )->second * arma::vectorise( arma::diagmat( back().v ) );
                                                else v = mTheta.find( trip<L,L,L>(l,front().l,back().l) )->second * arma::vectorise( back().v * front().v.t() ); }
    else cerr << "ERROR -- n-nary tree: " << *this << endl;
  }

  void sampleZ ( const map<L,arma::vec>& mPi, const map<trip<L,L,L>,arma::mat>& mTheta, mt19937& e, bool bRoot = true ) {
    if( bRoot )                                        { arma::vec post = mPi.find( l )->second % v; discrete_distribution<uint> d( post.begin(), post.end() ); z = d(e); }
    if( (size()==1 && front().size()>0) || size()==2 ) { arma::vec post = mTheta.find( trip<L,L,L>(l,front().l,(size()==1)?"-":back().l) )->second.row(z).t();
                                                         if (size()==1) post = post % arma::vectorise( arma::diagmat( back().v ) );
                                                         else post = post % arma::vectorise( back().v * front().v.t() );
                                                         discrete_distribution<uint> d( post.begin(), post.end() );
                                                         uint zz = d(e); front().z = zz/v.size(); back().z = zz%v.size();
                                                         for( auto& subtree : *this ) subtree.sampleZ( mPi, mTheta, e, false ); }
  }

  double getProb ( const map<L,arma::vec>& mPi, const map<trip<L,L,L>,arma::mat>& mTheta, map<L,arma::mat>& mXi, bool bRoot = true ) {
    double pr = 0.0;
    if( bRoot )                               pr = log( mPi.find( l )->second(z) );
    if(      size()==1 && front().size()==0 ) return pr + log( mXi.find( l )->second( z, W(front().l.getString().c_str()).toInt() ) );
    else if( size()==1 || size()==2 )         { pr += log( mTheta.find( trip<L,L,L>(l,front().l,(size()==1)?"-":back().l) )->second( z, v.size()*front().z + back().z ) );
                                                for( auto& subtree : *this ) pr += subtree.getProb( mPi, mTheta, mXi, false ); return pr; }
    cerr << "ERROR -- n-nary tree: " << *this << endl;
    return 0.0;
  }
};


////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  uint numTypes   = (nArgs>1) ? atoi(argv[1]) : 10;
  uint numIters   = (nArgs>2) ? atoi(argv[2]) : 100;
  uint numThreads = (nArgs>3) ? atoi(argv[3]) : 10;

  cerr << numTypes << " types, " << numIters << " iterations, " << numThreads << " threads." << endl;

  // Read trees...
  list<ForestTree> ltCorpus;
  while ( cin && EOF!=cin.peek() ) {
    ltCorpus.emplace_back(); ForestTree& t = ltCorpus.back();
    cin >> t >> "\n";
//    cout << "TREE: " << t << "\n";
  }

  cerr<<domW.getSize()<<" terminal tokens."<<endl;

  double pseudocount = 0.2;
  // Init root, nonterm, term counts...
  map<L,          arma::vec> cPi;
  map<trip<L,L,L>,arma::mat> cTheta;
  map<L,          arma::mat> cXi;
  for( auto& t : ltCorpus ) t.init( cPi, cTheta, cXi, numTypes );
  // Init root, nonterm, term models...
  map<L,          arma::vec> mPi;
  map<trip<L,L,L>,arma::mat> mTheta;
  map<L,          arma::mat> mXi;
  for( auto& t : ltCorpus ) t.init( mPi, mTheta, mXi, numTypes );

  mt19937 e( random_device{}() );

  // For each iteration...
  for( uint i=0; i<numIters; i++ ) {

    ////// I. Sample models given counts from trees...

    // Zero out counts...
    for( auto& lv   : cPi    ) cPi   [lv.first  ].zeros();
    for( auto& lllm : cTheta ) cTheta[lllm.first].zeros();
    for( auto& lm   : cXi    ) cXi   [lm.first  ].zeros();

    // Obtain counts...
    for( auto& t : ltCorpus ) t.addCounts( cPi, cTheta, cXi, numTypes );

    int totcounts = 0.0;
    for( auto& lv   : cPi    ) totcounts += arma::accu( cPi   [lv.first  ] );
    for( auto& lllm : cTheta ) totcounts += arma::accu( cTheta[lllm.first] );
    for( auto& lm   : cXi    ) totcounts += arma::accu( cXi   [lm.first  ] );

    // Update pi models...
    for( auto& lv : cPi ) {
      for( uint j=0; j<numTypes; j++ ) { gamma_distribution<double> d( pseudocount+lv.second(j), 1.0 ); mPi[lv.first](j) = d(e); }
      mPi[lv.first] = arma::normalise( mPi[lv.first], 1 );
    }

    // Update theta models...
    vector<thread> vtWorkers;
    for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {
      uint iCtr=0; for( auto& lllm : cTheta ) if( iCtr++%numThreads == numt ) {
        for( uint j=0; j<numTypes; j++ ) for( uint k=0; k<numTypes*numTypes; k++ )
          { gamma_distribution<double> d( pseudocount+lllm.second(j,k), 1.0 ); mTheta[lllm.first](j,k) = (lllm.first.third()==L("-") && k/numTypes!=k%numTypes) ? 0.0 : d(e); }
        mTheta[lllm.first] = arma::normalise( mTheta[lllm.first], 1, 1 );
      }
    }, numtglobal ));
    for( auto& w : vtWorkers ) w.join();

    // Update xi m(dels...
    vector<thread> vtWorkers2;
    for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers2.push_back( thread( [&] (uint numt) {
      uint iCtr=0; for( auto& lm : cXi ) if( iCtr++%numThreads == numt ) {
        for( uint j=0; j<numTypes; j++ ) for( uint k=0; k<uint(domW.getSize()); k++ ) { gamma_distribution<double> d( pseudocount+lm.second(j,k), 1.0 ); mXi[lm.first](j,k) = d(e); }
        mXi[lm.first] = arma::normalise( mXi[lm.first], 1, 1 );
      }
    }, numtglobal ));
    for( auto& w : vtWorkers2 ) w.join();

    double totmods = 0.0;
    for( auto& lv   : mPi    ) totmods += arma::accu( mPi   [lv.first  ] );
    for( auto& lllm : mTheta ) totmods += arma::accu( mTheta[lllm.first] );
    for( auto& lm   : mXi    ) totmods += arma::accu( mXi   [lm.first  ] );

    ////// II. Sample trees given models...

    // For each thread...
    vector<thread> vtWorkers3;
    for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers3.push_back( thread( [&] (uint numt) {
      mt19937 e( random_device{}() );
      // For each tree in process's modulo partition of corpus...
      uint itree=0; for( auto& t : ltCorpus ) if( itree++%numThreads == numt ) {
        // A. Calc forest from model, bottom-up...
        t.calcForest( mTheta, mXi );
        // B. Sample tree from forest, top-down...
        t.sampleZ( mPi, mTheta, e );
      }
    }, numtglobal ));
    for( auto& w : vtWorkers3 ) w.join();

    double lgpr = 0.0;
    for( auto& t : ltCorpus ) { lgpr += t.getProb(mPi,mTheta,mXi); }

    cerr << "Iteration " << i << ": totcounts=" << totcounts << " totmods=" << totmods << " logprob=" << lgpr << endl;
  }

  for( auto& t : ltCorpus ) cout << t << endl;

  /*
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
  */
}

