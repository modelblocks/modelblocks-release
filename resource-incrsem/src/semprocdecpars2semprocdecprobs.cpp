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
#include <thread>
#include <mutex>
using namespace std;
#include <armadillo>
using namespace arma;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
bool STORESTATE_TYPE = true;
bool STORESTATE_CHATTY = false;
uint FEATCONFIG = 0;
#include <StoreState.hpp>

////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc [] = " : ";
char psSpcEqualsSpc[] = " = ";

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<T> P;
typedef Delimited<T> A;
typedef Delimited<T> B;

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  // Define model structures...
  arma::mat matF;
  arma::mat matJ;
  map<PPredictor,map<P,double>> modP;
  map<W,list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>> lexW;
  map<APredictor,map<A,double>> modA;
  map<BPredictor,map<B,double>> modB;

  { // Define model lists...
    list<DelimitedTrip<psX,FPredictor,psSpcColonSpc,Delimited<FResponse>,psSpcEqualsSpc,Delimited<double>,psX>> lF;
    list<DelimitedTrip<psX,PPredictor,psSpcColonSpc,P,psSpcEqualsSpc,Delimited<double>,psX>> lP;
    list<DelimitedTrip<psX,WPredictor,psSpcColonSpc,W,psSpcEqualsSpc,Delimited<double>,psX>> lW;
    list<DelimitedTrip<psX,JPredictor,psSpcColonSpc,Delimited<JResponse>,psSpcEqualsSpc,Delimited<double>,psX>> lJ;
    list<DelimitedTrip<psX,APredictor,psSpcColonSpc,A,psSpcEqualsSpc,Delimited<double>,psX>> lA;
    list<DelimitedTrip<psX,BPredictor,psSpcColonSpc,B,psSpcEqualsSpc,Delimited<double>,psX>> lB;

    lA.emplace_back( DelimitedTrip<psX,APredictor,psSpcColonSpc,A,psSpcEqualsSpc,Delimited<double>,psX>(APredictor(1,0,1,'N','S',T("T"),T("-")),A("-"),1.0) );      // should be T("S")
    lB.emplace_back( DelimitedTrip<psX,BPredictor,psSpcColonSpc,B,psSpcEqualsSpc,Delimited<double>,psX>(BPredictor(1,0,1,'N','S','1',T("-"),T("S")),B("T"),1.0) );

    // For each command-line flag or model file...
    for ( int a=1; a<nArgs; a++ ) {
      if ( 0==strncmp(argv[a],"-f",2) ) FEATCONFIG = atoi(argv[a]+2);
      else {
        cerr << "Loading model " << argv[a] << "..." << endl;
        // Open file...
        ifstream fin (argv[a], ios::in );
        // Read model lists...
        int linenum = 0;
        while ( fin && EOF!=fin.peek() ) {
          if ( fin.peek()=='F' ) fin >> "F " >> *lF.emplace(lF.end()) >> "\n";
          if ( fin.peek()=='P' ) fin >> "P " >> *lP.emplace(lP.end()) >> "\n";
          if ( fin.peek()=='W' ) fin >> "W " >> *lW.emplace(lW.end()) >> "\n";
          if ( fin.peek()=='J' ) fin >> "J " >> *lJ.emplace(lJ.end()) >> "\n";
          if ( fin.peek()=='A' ) fin >> "A " >> *lA.emplace(lA.end()) >> "\n";
          if ( fin.peek()=='B' ) fin >> "B " >> *lB.emplace(lB.end()) >> "\n";
          if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
        }
        cerr << "Model " << argv[a] << " loaded." << endl;
      }
    }

    // Populate model structures...
    matF = arma::mat( FResponse::getDomain().getSize(), FPredictor::getDomainSize() );
    matJ = arma::mat( JResponse::getDomain().getSize(), JPredictor::getDomainSize() );
    for ( auto& prw : lF ) matF( prw.second().toInt(), prw.first().toInt() ) = prw.third();
    for ( auto& prw : lP ) modP[prw.first()][prw.second()] = prw.third();
    for ( auto& prw : lW ) lexW[prw.second()].emplace_back(prw.first(),prw.third());
    for ( auto& prw : lJ ) matJ( prw.second().toInt(), prw.first().toInt() ) = prw.third();
    for ( auto& prw : lA ) modA[prw.first()][prw.second()] = prw.third();
    for ( auto& prw : lB ) modB[prw.first()][prw.second()] = prw.third();
  }

  // Add unk...
  for( auto& entry : lexW ) {
    // for each word:{<category:prob>}
    for( auto& unklistelem : lexW[unkWord(entry.first.getString().c_str())] ) {
      // for each possible unked(word) category:prob pair
      bool BAIL = false;
      for( auto& listelem : entry.second ) {
        if (listelem.first == unklistelem.first) {
          BAIL = true;
          listelem.second = listelem.second + ( 0.000001 * unklistelem.second ); // merge actual likelihood and unk likelihood
        }
      }
      if (not BAIL) entry.second.push_back( DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>(unklistelem.first,0.000001*unklistelem.second) );
    }
  }

  cerr<<"Models ready."<<endl;

  DelimitedList<psX,DelimitedPair<psX,FPredictor,psEquals,Delimited<double>,psX>,psComma,psX> lpfpd;
  Delimited<FResponse> fr;
  arma::vec fresponses;
  PPredictor pp;  P p;

  // Read data...
  cerr << "Reading data...\n";
  while ( cin && EOF!=cin.peek() ) {
    //cerr<<char(cin.peek())<<endl;
    if( cin.peek()=='F' )      { lpfpd = DelimitedList<psX,DelimitedPair<psX,FPredictor,psEquals,Delimited<double>,psX>,psComma,psX> ( );
                                 cin  >> "F " >> lpfpd >> " : " >> fr >> "\n";
                                 fresponses = arma::zeros( matF.n_rows );
                                 for( auto& pfpd : lpfpd ) if( pfpd.first.toInt() < matF.n_cols ) fresponses += matF.col( pfpd.first.toInt() );
                                 fresponses = arma::exp( fresponses ); }
    else if( cin.peek()=='P' ) { cin  >> "P " >> pp    >> " : " >> p  >> "\n"; }
    else if( cin.peek()=='W' ) { WPredictor wp;  W w;
                                 cin  >> "W " >> wp    >> " : " >> w  >> "\n";
                                 pair<WPredictor,double> wppBest; FResponse frBest; double dBest=0.0;
                                 if( lexW.end()==lexW.find(w) ) w = unkWord(w.getString().c_str());
                                 for( auto& wpp : lexW[w] ) {
                                 //for( auto& wpp : (lexW.end()!=lexW.find(w)) ? lexW[w] : lexW[unkWord(w.getString().c_str())] ) {
                                   if( wpp.first.second == wp.second ) {
                                     for( FResponse fr1; fr1.toInt()<matF.n_rows; ++fr1 ) {
                                       if( fr1.getFork()==fr.getFork() && fr1.getE()==fr.getE() && fr1.getK()==wpp.first.first ) {
                                         double d = fresponses[fr1.toInt()]/arma::accu(fresponses);
                                         if( d * wpp.second > dBest ) { wppBest = wpp; frBest = fr1; dBest = d * wpp.second; }
                                       }
                                     }
                                   }
                                 }
                                 // what if no fr found?
                                 cout << "F " << lpfpd         << " : " << frBest << " = " << ((frBest.toInt()<matF.n_rows) ? fresponses[frBest.toInt()]/arma::accu(fresponses) : 0.0) << endl;
                                 cout << "P " << pp            << " : " << p      << " = " << modP[pp][p] << endl;
                                 cout << "W " << wppBest.first << " : " << w      << " = " << wppBest.second << endl;
    }
    else if( cin.peek()=='J' ) { DelimitedList<psX,DelimitedPair<psX,JPredictor,psEquals,Delimited<double>,psX>,psComma,psX> lpjpd;  Delimited<JResponse> jr;
                                 cin  >> "J " >> lpjpd >> " : " >> jr >> "\n";
                                 arma::vec jresponses = arma::zeros( matJ.n_rows );
                                 for( auto& pjpd : lpjpd ) if ( pjpd.first.toInt() < matJ.n_cols ) jresponses += matJ.col( pjpd.first.toInt() );
                                 jresponses = arma::exp( jresponses ); 
                                 cout << "J " << lpjpd << " : " << jr << " = " << ((jr.toInt()<matJ.n_rows) ? jresponses[jr.toInt()]/arma::accu(jresponses) : 0.0) << endl; }
    else if( cin.peek()=='A' ) { APredictor ap;  A a;
                                 cin  >> "A " >> ap    >> " : " >> a  >> "\n";
                                 cout << "A " << ap    << " : " << a  << " = " << modA[ap][a] << endl; }
    else if( cin.peek()=='B' ) { BPredictor bp;  B b;
                                 cin  >> "B " >> bp    >> " : " >> b  >> "\n";
                                 cout << "B " << bp    << " : " << b  << " = " << modB[bp][b] << endl;}
    else { string line; getline(cin,line); cout<<line<<endl; }
  }
}

