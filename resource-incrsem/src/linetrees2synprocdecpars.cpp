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

#define ARMA_64BIT_WORD
#include <iostream>
#include <fstream>
#include <list>
#include <regex>
using namespace std;
#include <armadillo>
using namespace arma;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
bool STORESTATE_TYPE = false;
#include <StoreStateSynProc.hpp>
#include <Tree.hpp>

map<L,double> mldLemmaCounts;
const int MINCOUNTS = 100;

////////////////////////////////////////////////////////////////////////////////

int getArityGivenLabel ( const L& l ) {
  int depth = 0;
  int arity = 0;
  if ( l[0]=='N' ) arity++;
  for ( uint i=0; i<l.size(); i++ ) {
    if ( l[i]=='{' ) depth++;
    if ( l[i]=='}' ) depth--;
    if ( l[i]=='-' && l[i+1]!='l' && l[i+1]!='x' && depth==0 ) arity++;
  }
  return arity;
}

////////////////////////////////////////////////////////////////////////////////

T T_COLON ( "Pk" );                       // must be changed to avoid confusion with " : " delimiter in P params (where type occurs individually).
T T_CONTAINS_COMMA ( "!containscomma!" ); // must be changed to avoid confusion with "," delimiter in F,J params.

T getType ( const L& l ) {
/* COMMENTED OUT FOR COMPATIBILITY WITH BERK MODEL
  if ( l[0]==':' )                 return T_COLON;
  if ( l.find(',')!=string::npos ) return T_CONTAINS_COMMA;
  return string( string( l, 0, l.find("-l") ), 0, l.find("-xX") ).c_str();
*/
  return l.c_str();
}

////////////////////////////////////////////////////////////////////////////////

void calcContext ( const Tree& tr, int s=1, int d=0 ) {
  static F          f;
  static Sign       aPretrm;
  static StoreState q;

  // At unary preterminal...
  if ( tr.size()==1 && tr.front().size()==0 ) {

    f            = 1 - s;
    T t          = getType( L(tr) );
    aPretrm      = t;

    // Print preterminal / fork-phase predictors...
    cout << "F " << q.calcForkTypeCondition()    << " : " << f             << endl;
    cout << "P " << q.calcPretrmTypeCondition(f) << " : " << t             << endl;
    cout << "W " << t                            << " : " << L(tr.front()) << endl;
  }

  // At unary nonpreterminal...
  else if ( tr.size()==1 ) {
    calcContext ( tr.front(), s, d );
  }

  // At binary nonterminal...
  else if ( tr.size()==2 ) {

    // Traverse left child...
    calcContext ( tr.front(), 0, d+s );

    J j = s;

    cout << "J " << q.calcJoinTypeCondition(f,aPretrm)                  << " : " << j                   << endl;
    cout << "A " << q.calcApexTypeCondition(f,j,aPretrm)                << " : " << getType(tr)         << endl;
    cout << "B " << q.calcBrinkTypeCondition(f,j,getType(tr),aPretrm)   << " : " << getType(tr.back())  << endl;

    // Update storestate...
    q = StoreState ( q, f, j, getType(tr), getType(tr.back()), aPretrm );

    // Traverse right child...
    calcContext ( tr.back(), 1, d );
  }

  // At abrupt terminal (e.g. 'T' discourse)...
  else if ( tr.size()==0 );

  else cerr<<"ERROR: non-binary non-unary-preterminal: " << tr << endl;
}

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  list<DelimitedPair<psX,Delimited<double>,psSpace,L,psX>> lLC;

  // For each command-line flag or model file...
  for ( int a=1; a<nArgs; a++ ) {
    if      ( 0==strcmp(argv[a],"-d") ) NODEP = true;
    else if ( 0==strcmp(argv[a],"t") ) STORESTATE_TYPE = true;
    else {
      cerr << "Loading model " << argv[a] << "..." << endl;
      // Open file...
      ifstream fin (argv[a], ios::in );
      // Read model lists...
      int linenum = 0;
      while ( fin && EOF!=fin.peek() ) {
        fin >> *lLC.emplace(lLC.end()) >> "\n";
        if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
      }
      cerr << "Model " << argv[a] << " loaded." << endl;
      for ( auto& l : lLC ) mldLemmaCounts[l.second] = l.first;
    }
  }

  while ( cin && EOF!=cin.peek() ) {
    //Tree t;
    Tree t("T"); t.emplace_back(); t.emplace_back("T");
    cin >> t.front() >> "\n";
    cout << "TREE: " << t << "\n";
    if ( t.front().size() > 0 ) calcContext ( t );
  }
}


