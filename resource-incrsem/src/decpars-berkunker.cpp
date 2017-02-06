
#include <random>
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <Delimited.hpp>
#include <BerkUnkWord.hpp>

DiscreteDomain<int> domC;
class C : public Delimited<DiscreteDomainRV<int,domC>> {
 public:
  C ( )                : Delimited<DiscreteDomainRV<int,domC>> ( )    { }
  C ( int i )          : Delimited<DiscreteDomainRV<int,domC>> ( i )  { }
  C ( const char* ps ) : Delimited<DiscreteDomainRV<int,domC>> ( ps ) { }
};

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  map<string,map<string,double>> remainder;
  map<string,double> normL;
  map<W,double> wordcount;
  map<C,map<W,double>> lex;

  // Read in pcfg model...
  int linenum = 0;
  while ( cin && EOF!=cin.peek() ) {
    C c;
    W w;
    Delimited<string> l1, l2;
    if( cin.peek()!='W' ) { cin >> l1 >> " : " >> l2 >> "\n"; remainder[l1][l2] += 1; normL[l1] += 1; }
    if( cin.peek()=='W' ) { cin >> "X "  >> c >> " : " >> w  >> "\n";  lex[c][w] += 1;  wordcount[w] += 1; }
    if( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
  }

  static std::default_random_engine         e{};
  static std::uniform_int_distribution<int> d{0, 1};

  for( const auto& llp : remainder )
    for( const auto& lp : llp.second )
      for( int i=0; i<lp.second; i++ )
        cout << llp.first << " : " << lp.first << endl;

  // Randomly replace singleton words with unks...
  map<C,map<W,double>> unks;
  map<C,double> norm;
  for( const auto& cwp : lex ) {
    for( const auto& wp : cwp.second ) {
      if( 1 == wordcount[wp.first] && 1 == d(e) ) {
        unks[cwp.first][unkWord(wp.first.getString().c_str())] += 0.000001 * wp.second;
        norm[cwp.first] += 1.000001 * wp.second;
      }
      else {
        norm[cwp.first] += wp.second;
      }
    }
  }

  // Print updated lexicon...
  for( const auto& cwp : lex ) {
    for( const auto& wp : cwp.second ) {
      // Print each word as often as it occurs...
      for( int i=0; i<wp.second; i++ ) {
        cout << "W " << cwp.first << " : " << wp.first << endl;
      }
    }
  }
  for( const auto& cwp : unks ) {
    for( const auto& wp : cwp.second ) {
      // NOTE: print each unk only once (each unk is unique)...
      cout << "W " << cwp.first << " : " << wp.first << endl;
    }
  }
}
