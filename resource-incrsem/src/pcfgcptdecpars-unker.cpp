
#include <random>
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <Delimited.hpp>
#include <BerkUnkWord.hpp>
#include <RandoUnkWord.hpp>

DiscreteDomain<int> domC;
class C : public Delimited<DiscreteDomainRV<int,domC>> {
 public:
  C ( )                : Delimited<DiscreteDomainRV<int,domC>> ( )    { }
  C ( int i )          : Delimited<DiscreteDomainRV<int,domC>> ( i )  { }
  C ( const char* ps ) : Delimited<DiscreteDomainRV<int,domC>> ( ps ) { }
};

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  bool bBerkUnker = ( nArgs>1 && argv[1][0]=='-' && argv[1][1]=='u' );

  map<string,map<string,double>> gram;
  map<string,double> normL;
  map<W,double> wordcount;
  map<C,map<W,double>> lex;

  // Read in pcfg model...
  int linenum = 0;
  while ( cin && EOF!=cin.peek() ) {
    C c;
    W w;
    Delimited<string> l1, l2;
    if( cin.peek()=='C' ) { cin >> l1 >> " : " >> l2 >> "\n"; gram[l1][l2] += 1; normL[l1] += 1; }
    if( cin.peek()=='X' ) { cin >> "X "  >> c >> " : " >> w  >> "\n";  lex[c][w] += 1;  wordcount[w] += 1; }
    if( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
  }

  static std::default_random_engine         e{};
  static std::uniform_int_distribution<int> d{0, 1};

  for( const auto& llp : gram )
    for( const auto& lp : llp.second )
      cout << llp.first << " : " << lp.first << " = " << lp.second/normL[llp.first] << endl;

  // Randomly replace singleton words with unks...
  map<C,map<W,double>> unks;
  map<C,double> norm;
  for( const auto& cwp : lex ) {
    for( const auto& wp : cwp.second ) {
      if( 1 == wordcount[wp.first] && 1 == d(e) ) {
        unks[cwp.first][(bBerkUnker) ? unkWordBerk(wp.first.getString().c_str()) : unkWord(wp.first.getString().c_str())] += 0.000001 * wp.second;
        norm[cwp.first] += 1.000001 * wp.second;
      }
      else {
        norm[cwp.first] += wp.second;
//        for( int i=0; i<wp.second; i++ ) {
//          cout << "X " << cwp.first << " : " << wp.first << endl;
//        }
      }
    }
  }

  // Print updated lexicon...
  for( const auto& cwp : lex ) {
    for( const auto& wp : cwp.second ) {
      cout << "X " << cwp.first << " : " << wp.first << " = " << wp.second/norm[cwp.first] << endl;
    }
  }
  for( const auto& cwp : unks ) {
    for( const auto& wp : cwp.second ) {
      //for( int i=0; i<wp.second; i++ ) {
        cout << "X " << cwp.first << " : " << wp.first << " = " << wp.second/norm[cwp.first] << endl;
      //}
    }
  }
}
