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

//using namespace std;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <Delimited.hpp>

////////////////////////////////////////////////////////////////////////////////

const int MAXITERS = 20;

const int MAXDEPTH = 4;

// Initialized double...
class Double {
 private:
  double val;
 public:
  Double ( )                 : val(0.0)      { }
  Double ( const double& t ) : val(t)        { }
  Double ( const char* ps )  : val(stod(ps)) { }
  operator double() const { return val; }
  Double operator+= ( const double& t ) { return val+=t; }
  Double operator-= ( const double& t ) { return val-=t; }
  Double operator*= ( const double& t ) { return val*=t; }
  Double operator/= ( const double& t ) { return val/=t; }
};

typedef char S;

typedef int D;

typedef int F;

typedef int J;

DiscreteDomain<int> domC;
class C : public DiscreteDomainRV<int,domC> {
 public:
  C ( )                : DiscreteDomainRV<int,domC> ( )    { }
  C ( int i )          : DiscreteDomainRV<int,domC> ( i )  { }
  C ( const char* ps ) : DiscreteDomainRV<int,domC> ( ps ) { }
};

typedef C A;

typedef C B;

typedef C P;

DiscreteDomain<int> domW;
class W : public DiscreteDomainRV<int,domW> {
 public:
  W ( )                : DiscreteDomainRV<int,domW> ( )    { }
  W ( int i )          : DiscreteDomainRV<int,domW> ( i )  { }
  W ( const char* ps ) : DiscreteDomainRV<int,domW> ( ps ) { }
};

template<class M>
void normalize( M& mcmvp ) {
  for( auto& cmvp : mcmvp ) {
    double dTot = 0.0;
    for( auto& vp : cmvp.second )
      dTot += vp.second;
    for( auto& vp : cmvp.second )
      vp.second /= dTot;
  }
}

template<class M>
void write( char c, M& mcmvp ) {
  for( auto& cmvp : mcmvp ) {
    for( auto& vp : cmvp.second )
      cout << c << " " << cmvp.first << " : " << vp.first << " = " << vp.second << endl;
  }
}

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  typedef DelimitedPair<psX,Delimited<C>,psSpace,Delimited<C>,psX> CC;
  map<C,map<CC,double>> pcfg;
  map<C,map<W,double>> lex;

  // Read in CC model...
//  // For each command-line flag or model file...
//  for ( int a=1; a<nArgs; a++ ) {
//    if      ( 0==strcmp(argv[a],"-v") ) VERBOSE = 1;
//    else if ( 0==strcmp(argv[a],"-V") ) VERBOSE = 2;
//    else if ( 0==strncmp(argv[a],"-b",2) ) BEAM_WIDTH = atoi(argv[a]+2);
//    //else if ( string(argv[a]) == "t" ) STORESTATE_TYPE = true;
//    else {
//      cerr << "Loading model " << argv[a] << "..." << endl;
//      // Open file...
//      ifstream fin (argv[a], ios::in );
//      // Read model lists...
      int linenum = 0;
      while ( cin && EOF!=cin.peek() ) {
        Delimited<C> c;
        CC cc;
        Delimited<W> w;
        Delimited<double> pr;
        if( cin.peek()=='C' ) { cin >> "CC " >> c >> " : " >> cc >> " = " >> pr >> "\n";  pcfg[c][cc]=pr; }
        if( cin.peek()=='X' ) { cin >> "X "  >> c >> " : " >> w  >> " = " >> pr >> "\n";  lex [c][w] =pr; }
        if( cin.peek()=='R' ) { cin >> "R : "  >> c >> " = " >> pr >> "\n";  pcfg[C("T")][CC(c,C("T"))]=pr; }
        //cerr << c << "    " << cc << endl;
        //if ( fin.peek()=='F' ) fin >> "F " >> *lF.emplace(lF.end()) >> "\n";
        //if ( fin.peek()=='P' ) fin >> "P " >> *lP.emplace(lP.end()) >> "\n";
        //if ( fin.peek()=='W' ) fin >> "W " >> *lW.emplace(lW.end()) >> "\n";
        //if ( fin.peek()=='J' ) fin >> "J " >> *lJ.emplace(lJ.end()) >> "\n";
        //if ( fin.peek()=='A' ) fin >> "A " >> *lA.emplace(lA.end()) >> "\n";
        //if ( fin.peek()=='B' ) fin >> "B " >> *lB.emplace(lB.end()) >> "\n";
        if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
      }
//      cerr << "Model " << argv[a] << " loaded." << endl;
//    }
//  }

  //write( 'G', pcfg );

  cerr << "Calc delta model..." << endl;

  // Calc delta model...
  typedef DelimitedTrip<psX,D,psSpace,S,psSpace,Delimited<C>,psX> DSC;
  map<DSC,Double> deltas[2];
  for( int i=0; i<=MAXITERS; i++ ) {
    deltas[i%2] = map<DSC,Double> ( );
    for( D d=0; d<=MAXDEPTH+1; d++ )
      for( const S& s : {'L','R'} ) if( d<=MAXDEPTH || s=='L' )
        for( const auto& cmccd : pcfg )
          for( const auto& ccd : cmccd.second ) {
            const C& c = cmccd.first;
            const C& c0 = ccd.first.first;
            const C& c1 = ccd.first.second;
            const double& pr = ccd.second;
            const double  pr2 = (i==0) ? 0.0 : deltas[(i-1)%2][DSC((s=='L')?d:d+1,'L',c0)] * deltas[(i-1)%2][DSC(d,'R',c1)];
            if ( c1==C("-") && pr>0.0 ) deltas[i%2][DSC(d,s,c)] += pr;
            else if ( i>0 && pr>0.0 && pr2>0.0 ) deltas[i%2][DSC(d,s,c)] += pr * pr2; 
          }
  }
  map<DSC,Double> delta = deltas[MAXITERS%2];
  delta[DSC(0,'R',C("T"))]=1.0;  // Needed bc T -> S T rule never grounds out.

  //for( auto& dscd : delta )
  //  cout << "d " << dscd.first << " = " << dscd.second << endl;

  cerr << "Calc gamma model..." << endl;

  // Calc gamma model...
  map<DSC,map<CC,Double>> gamma;
  for( D d=0; d<=MAXDEPTH+1; d++ )
    for( const S& s : {'L','R'} ) if( d<=MAXDEPTH || s=='L' )
      for( const auto& cccd : pcfg ) {
        const C& c = cccd.first;
        if( (d==0 && s=='R' && c==C("T")) || (d!=0 && c!=C("T")) )
          for( const auto& ccd : cccd.second ) {
            const C& c0 = ccd.first.first;
            const C& c1 = ccd.first.second;
            double pr = ccd.second;
            if( c1!=C("-") && delta[DSC(d,s,c)]>0.0 ) pr *= delta[DSC((s=='L')?d:d+1,'L',c0)] * delta[DSC(d,'R',c1)] / delta[DSC(d,s,c)];
            if( pr>0.0 ) gamma[DSC(d,s,c)][ccd.first] += pr;
          }
      }

  //write( 'g', gamma );

  cerr << "Calc star models..." << endl;

  // Calc zero,plus,star models...
  typedef DelimitedPair<psX,D,psSpace,Delimited<B>,psX> DB;
  map<DB,map<C,double>> plus_iter[2];
  map<DB,map<C,double>> zero;
  map<DB,map<C,double>> plus;
  map<DB,map<C,double>> star;
  for( int i=0; i<=MAXITERS; i++ ) {
    plus_iter[i%2] = map<DB,map<C,double>> ( );
    cerr << "  i=" << i << "/" << MAXITERS << endl;
    if( i==0 ) {
      for( int d=0; d<=MAXDEPTH; d++ ) {
        for( const auto& bmccd : pcfg ) {
          const C& b = bmccd.first;
          if( (d==0 && b==C("T")) || (d!=0 && b!=C("T")) ) {
            zero[DB(d,b)][b] += 1.0;
            star[DB(d,b)][b] += 1.0;
            for( const auto& ccd : gamma[DSC(d,'R',b)] ) {
              if( ccd.first.second!=C("-") ) {
                const C& c0 = ccd.first.first;
                plus_iter[i%2][DB(d,b)][c0] += ccd.second;
                plus[DB(d,b)][c0]           += ccd.second;
                star[DB(d,b)][c0]           += ccd.second;
              }
            }
          }
        }
      }
    }
    else {
      for( const auto& dbmcd : plus_iter[(i-1)%2] ) {
        const D& d = dbmcd.first.first;
        for( const auto& cd : dbmcd.second ) {
          for( const auto& ccd : gamma[DSC(d+1,'L',cd.first)] ) {
            if( ccd.first.second!=C("-") ) { 
              const C& c0 = ccd.first.first;
              plus_iter[i%2][dbmcd.first][c0] += cd.second * ccd.second;
              plus[dbmcd.first][c0]           += cd.second * ccd.second;
              star[dbmcd.first][c0]           += cd.second * ccd.second;
            }
          }
        }
      }
    }
  }

  //write( 'z', zero );
  //write( 'p', plus );
  //write( 's', star );

  cerr << "Calc F models..." << endl;

  // Calc F,P...
  map<DelimitedPair<psX,D,psSpace,B,psX>,map<F,Double>> mF;
  map<DelimitedTrip<psX,D,psSpace,F,psSpace,B,psX>,map<P,Double>> mP;
  for( const auto& dbmpd : plus ) {
    const D& d = dbmpd.first.first;
    const B& b = dbmpd.first.second;
    for( const auto& pd : dbmpd.second ) {
      const P& p = pd.first;
      for( const auto& wxd : gamma[DSC(d+1,'L',p)] )
        if( wxd.first.second==C("-") ) {
          mF[DelimitedPair<psX,D,psSpace,B,psX>(d,b)][1]             += plus[dbmpd.first][p] * wxd.second;
          mP[DelimitedTrip<psX,D,psSpace,F,psSpace,B,psX>(d,1,b)][p] += plus[dbmpd.first][p] * wxd.second;
        }
    }
  }
  for( const auto& dbmpd : zero ) {
    const D& d = dbmpd.first.first;
    const B& b = dbmpd.first.second;
    for( const auto& pd : dbmpd.second ) {
      const P& p = pd.first;
      for( const auto& wxd : gamma[DSC(d,'R',p)] )
        if( wxd.first.second==C("-") ) {
          mF[DelimitedPair<psX,D,psSpace,B,psX>(d,b)][0]             += zero[dbmpd.first][p] * wxd.second;
          mP[DelimitedTrip<psX,D,psSpace,F,psSpace,B,psX>(d,0,b)][p] += zero[dbmpd.first][p] * wxd.second;
        }
    }
  }
  normalize( mF );  write( 'F', mF );
  normalize( mP );  write( 'P', mP );
  write( 'W', lex );

  cerr << "Calc J models..." << endl;

  // Calc J,A,B...
  map<DelimitedTrip<psX,D,psSpace,B,psSpace,C,psX>,map<J,Double>> mJ;
  map<DelimitedQuad<psX,D,psSpace,J,psSpace,B,psSpace,C,psX>,map<A,Double>> mA;
  map<DelimitedQuad<psX,D,psSpace,J,psSpace,A,psSpace,C,psX>,map<B,Double>> mB;
  for( const auto& dbmad : plus ) {
    const D& d = dbmad.first.first;
    const B& b = dbmad.first.second;
    for( const auto& ad : dbmad.second ) {
      const A& a = ad.first;
      for( const auto& ccp : gamma[DSC(d+1,'L',a)] )
        if( ccp.first.second != C("-") && plus[dbmad.first][a]>0.0 ) {
          const C& c0 = ccp.first.first;
          const C& c1 = ccp.first.second;
          mJ[DelimitedTrip<psX,D,psSpace,B,psSpace,C,psX>(d+1,b,c0)][0]              += plus[dbmad.first][a] * ccp.second;
          mA[DelimitedQuad<psX,D,psSpace,J,psSpace,B,psSpace,C,psX>(d+1,0,b,c0)][a]  += plus[dbmad.first][a] * ccp.second;
          mB[DelimitedQuad<psX,D,psSpace,J,psSpace,A,psSpace,C,psX>(d+1,0,a,c0)][c1] += plus[dbmad.first][a] * ccp.second;
        }
    }
  }
  for( const auto& dbmad : zero ) {
    const D& d = dbmad.first.first;
    const B& b = dbmad.first.second;
    for( const auto& ad : dbmad.second ) {
      const A& a = ad.first;
      for( const auto& ccp : gamma[DSC(d,'R',a)] )
        if( ccp.first.second != C("-") && zero[dbmad.first][a]>0.0 ) {
          const C& c0 = ccp.first.first;
          const C& c1 = ccp.first.second;
          mJ[DelimitedTrip<psX,D,psSpace,B,psSpace,C,psX>(d+1,b,c0)][1]               += zero[dbmad.first][a] * ccp.second;
          mA[DelimitedQuad<psX,D,psSpace,J,psSpace,B,psSpace,C,psX>(d,1,b,C("-"))][a] += zero[dbmad.first][a] * ccp.second;
          mB[DelimitedQuad<psX,D,psSpace,J,psSpace,A,psSpace,C,psX>(d,1,a,c0)][c1]    += zero[dbmad.first][a] * ccp.second;
        }
    }
  }
  normalize( mJ );  write( 'J', mJ );
  normalize( mA );  write( 'A', mA );
  normalize( mB );  write( 'B', mB );
}


