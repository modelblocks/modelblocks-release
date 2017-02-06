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
#include <algorithm>
using namespace std;
#include <armadillo>
using namespace arma;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
bool STORESTATE_TYPE = true;
#include <StoreStateFJABPForBerk.hpp>

uint BEAM_WIDTH = 1000;
uint VERBOSE    = 0;

////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";

////////////////////////////////////////////////////////////////////////////////

typedef T P;
typedef T A;
typedef T B;

////////////////////////////////////////////////////////////////////////////////

class BeamElement : public DelimitedSext<psX,Delimited<double>,psSpace,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psSpace,Delimited<int>,psX> {
 public:
  BeamElement ( )                                                               : DelimitedSext<psX,Delimited<double>,psSpace,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psSpace,Delimited<int>,psX>()            { }
  BeamElement ( double d, const Sign& a, F f, J j, const StoreState& q, int i ) : DelimitedSext<psX,Delimited<double>,psSpace,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psSpace,Delimited<int>,psX>(d,a,f,j,q,i) { }
};

class Beam : public DelimitedVector<psX,BeamElement,psLine,psX> {
 public:
  Beam ( )        : DelimitedVector<psX,BeamElement,psLine,psX>() { }
  //Beam ( uint i ) : priority_queue<BeamElement,DelimitedVector<psX,BeamElement,psLine,psX>,greater<BeamElement>>() { reserve(i); }
  //iterator<BeamElement>& begin() { return c.begin(); }
  //iterator<BeamElement>& end() { return c.end(); }
};

class Trellis : public vector<Beam> {
 private:
  DelimitedList<psX,BeamElement,psLine,psX> lbe;
 public:
  Trellis ( ) : vector<Beam>() { reserve(100); }
  Beam& operator[] ( uint i ) { if ( i==size() ) emplace_back(); return vector<Beam>::operator[](i); }
  const DelimitedList<psX,BeamElement,psLine,psX>& getMostLikelySequence ( ) {
    lbe.clear();  if( back().size()>0 ) lbe.push_front( back().front() );
    //for( BeamElement& be : back() )  /*if( be.fifth().size()==0 )*/ { lbe.push_front( be ); break; }
    if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( at(t).at(lbe.front().sixth()) );
//    if( lbe.size()>0 ) lbe.emplace_back();
    return lbe;
  }
};

class StreamTrellis : public vector<Beam> {
 public:
  StreamTrellis ( ) : vector<Beam>(2) { }       // previous and next beam.
  Beam&       operator[] ( uint i )       { return vector<Beam>::operator[](i%2); }
  const Beam& operator[] ( uint i ) const { return vector<Beam>::operator[](i%2); }
};


////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  // Define model structures...
  map<FPredictor,map<F,double>> modF;
  map<PPredictor,map<P,double>> modP;
  map<W,list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>> lexW;
  map<JPredictor,map<J,double>> modJ;
  map<APredictor,map<A,double>> modA;
  map<BPredictor,map<B,double>> modB;

  { // Define model lists...
    list<DelimitedTrip<psX,FPredictor,psSpcColonSpc,F,psSpcEqualsSpc,Delimited<double>,psX>> lF;
    list<DelimitedTrip<psX,PPredictor,psSpcColonSpc,P,psSpcEqualsSpc,Delimited<double>,psX>> lP;
    list<DelimitedTrip<psX,WPredictor,psSpcColonSpc,W,psSpcEqualsSpc,Delimited<double>,psX>> lW;
    list<DelimitedTrip<psX,JPredictor,psSpcColonSpc,J,psSpcEqualsSpc,Delimited<double>,psX>> lJ;
    list<DelimitedTrip<psX,APredictor,psSpcColonSpc,A,psSpcEqualsSpc,Delimited<double>,psX>> lA;
    list<DelimitedTrip<psX,BPredictor,psSpcColonSpc,B,psSpcEqualsSpc,Delimited<double>,psX>> lB;

    lF.emplace_back( DelimitedTrip<psX,FPredictor,psSpcColonSpc,F,psSpcEqualsSpc,Delimited<double>,psX>(FPredictor(1,T("."),T(".")),0,1.0) );
    lJ.emplace_back( DelimitedTrip<psX,JPredictor,psSpcColonSpc,J,psSpcEqualsSpc,Delimited<double>,psX>(JPredictor(1,T("T"),T("S")),1,1.0) );
    lA.emplace_back( DelimitedTrip<psX,APredictor,psSpcColonSpc,A,psSpcEqualsSpc,Delimited<double>,psX>(APredictor(0,1,T("T"),T("-")),A("-"),1.0) );      // should be T("S")
    lB.emplace_back( DelimitedTrip<psX,BPredictor,psSpcColonSpc,B,psSpcEqualsSpc,Delimited<double>,psX>(BPredictor(0,1,T("-"),T("S")),B("T"),1.0) );

    // For each command-line flag or model file...
    for( int a=1; a<nArgs; a++ ) {
      if      ( 0==strcmp(argv[a],"-v") ) VERBOSE = 1;
      else if ( 0==strcmp(argv[a],"-V") ) VERBOSE = 2;
      else if ( 0==strncmp(argv[a],"-b",2) ) BEAM_WIDTH = atoi(argv[a]+2);
      //else if ( string(argv[a]) == "t" ) STORESTATE_TYPE = true;
      else {
        cerr << "Loading model " << argv[a] << "..." << endl;
        // Open file...
        ifstream fin (argv[a], ios::in );
        // Read model lists...
        int linenum = 0;
        while ( fin && EOF!=fin.peek() ) {
          if( fin.peek()=='F' ) fin >> "F " >> *lF.emplace(lF.end()) >> "\n";
          if( fin.peek()=='P' ) fin >> "P " >> *lP.emplace(lP.end()) >> "\n";
          if( fin.peek()=='W' ) fin >> "W " >> *lW.emplace(lW.end()) >> "\n";
          if( fin.peek()=='J' ) fin >> "J " >> *lJ.emplace(lJ.end()) >> "\n";
          if( fin.peek()=='A' ) fin >> "A " >> *lA.emplace(lA.end()) >> "\n";
          if( fin.peek()=='B' ) fin >> "B " >> *lB.emplace(lB.end()) >> "\n";
          if( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
        }
        cerr << "Model " << argv[a] << " loaded." << endl;
      }
    }

    // Populate model structures...
    for( auto& prw : lF ) modF[prw.first()][prw.second()] = prw.third();
    for( auto& prw : lP ) modP[prw.first()][prw.second()] = prw.third();
    for( auto& prw : lW ) lexW[prw.second()].emplace_back(prw.first(),prw.third());
    for( auto& prw : lJ ) modJ[prw.first()][prw.second()] = prw.third();
    for( auto& prw : lA ) modA[prw.first()][prw.second()] = prw.third();
    for( auto& prw : lB ) modB[prw.first()][prw.second()] = prw.third();
  }

  cerr<<"Models ready."<<endl;

  // Add unk...
  for( auto& entry : lexW ){
    // for each word:{<category:prob>}
    for( auto& unklistelem : lexW[unkWord(entry.first.getString().c_str())] ){
      // for each possible unked(word) category:prob pair
      bool BAIL = false;
      for( auto& listelem : entry.second ) {
        if (listelem.first == unklistelem.first) {
          BAIL = true;
          listelem.second = listelem.second + unklistelem.second; // merge actual likelihood and unk likelihood
        }
      }
      if (not BAIL) entry.second.push_back(unklistelem);
    }
  }

  // For each line in stdin...
  for( int linenum=1; cin && EOF!=cin.peek(); linenum++ ) {

    Trellis                                beams;  // sequence of beams
    uint                                   t=0;    // time step
    DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // input list

    // Allocate space in beams to avoid reallocation...
    for( auto& b : beams ) b.reserve(BEAM_WIDTH);
    // Create initial beam element...
    StoreState ssInit; ssInit.emplace_back(Sign("S"),Sign("."));  //Sign("S-lS^g_0"),Sign("._0"));
    beams[0].emplace_back(0.0,Sign("."),0,1,ssInit,0);  //Sign("._0"),0,1,ssInit,0);
    // Read sentence...
    cin >> lwSent >> "\n";
    // Add first word to end of sentence...
    lwSent.emplace_back( lwSent.front() );
    cerr << "#" << linenum;

    // For each word...
    for( auto& w_t : lwSent ) {

      cerr << " " << w_t;
      if( VERBOSE ) cout << "WORD:" << w_t << endl;

      // Create beam for current time step...
      beams[++t].clear();

      // For each hypothesized storestate at previous time step...
      for( auto& be_tdec1 : beams[t-1] ) {
        double            lgpr_tdec1 = be_tdec1.first();  // prob of prev storestate
        const StoreState& q_tdec1    = be_tdec1.fifth();  // prev storestate
        const Sign&       p_tdec1    = be_tdec1.second(); // prev P
        //F                 f_tdec1    = be_tdec1.third();  // prev F
        //J                 j_tdec1    = be_tdec1.fourth(); // prev J

        if( VERBOSE>1 ) cout << "  from (" << be_tdec1.fifth() << ")" << endl;

        // For each possible no-fork or fork decision...
        FPredictor fpredictor = q_tdec1.calcForkTypeCondition( p_tdec1 );
        if( VERBOSE>1 ) cout << "    F " << fpredictor << "..." << endl;
        for( auto& tfp : modF[fpredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tfp.second) > beams[t].front().first() ) {
          const F& f = tfp.first;
          if( VERBOSE>1 ) cout << "      F " << fpredictor << " : " << tfp.first << " = " << tfp.second << endl;

          // For each possible no-join or join decision...
          JPredictor jpredictor = q_tdec1.calcJoinTypeCondition( f, p_tdec1 );
          if( VERBOSE>1 ) cout << "        J " << jpredictor << "..." << endl;
          for( auto& tjp : modJ[jpredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tfp.second) + log(tjp.second) > beams[t].front().first() ) {
            const J& j = tjp.first;
            if( VERBOSE>1 ) cout << "          J " << jpredictor << " : " << tjp.first << " = " << tjp.second << endl;

            // For each possible apex category label...
            APredictor apredictor = q_tdec1.calcApexTypeCondition( f, j, p_tdec1 );  // save apredictor for use in prob calc
            if( VERBOSE>1 ) cout << "            A " << apredictor << "..." << endl;
            for( auto& tpA : modA[apredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tfp.second) + log(tjp.second) + log(tpA.second) > beams[t].front().first() ) {

              if( VERBOSE>1 ) cout << "            A " << apredictor << " : " << tpA.first << " = " << tpA.second << endl;

              // For each possible brink category label...
              BPredictor bpredictor = q_tdec1.calcBrinkTypeCondition( f, j, tpA.first, p_tdec1 );  // bpredictor for prob calc
              if( VERBOSE>1 ) cout << "              B " << bpredictor << "..." << endl;
              for( auto& tpB : modB[bpredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tfp.second) + log(tjp.second) + log(tpA.second) + log(tpB.second) > beams[t].front().first() ) {

                if( VERBOSE>1 ) cout << "              B " << bpredictor << " : " << tpB.first << " = " << tpB.second << endl;

                StoreState ss( q_tdec1, f, j, tpA.first, tpB.first, p_tdec1 );
                // For each possible lemma (context + label + prob) for preterminal of current word...
                for( auto& ktpr_p_t : (lexW.end()!=lexW.find(w_t)) ? lexW[w_t] : lexW[unkWord(w_t.getString().c_str())] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tfp.second) + log(tjp.second) + log(tpA.second) + log(tpB.second) + log(ktpr_p_t.second) > beams[t].front().first() ) {
                  T t_p_t           = ktpr_p_t.first;  // label of cunnent preterminal
                  double probwgivkl = ktpr_p_t.second; // probability of current word given current preterminal
                  //const Sign& aAncstr = q_tdec1.getAncstr( f );                                        // aAncstr (brink of previous incomplete cat
                  Sign aPretrm = t_p_t;                                                                // aPretrm (pos tag)

                  if( VERBOSE>1 ) cout << "                W " << t_p_t << " : " << w_t << " = " << probwgivkl << endl;

                  // If preterminal prob is nonzero... 
                  PPredictor ppredictor = ss.calcPretrmTypeCondition();
                  if( VERBOSE>1 ) cout << "                    P " << ppredictor << "..." << endl;
                  if( modP[ppredictor].end()!=modP[ppredictor].find(t_p_t) ) {

                    if( VERBOSE>1 ) cout << "                    P " << ppredictor << " : " << t_p_t << " = " << modP[ppredictor][t_p_t] << endl;

                    // Calculate probability and storestate and add to beam...
                    //double probJoin = tjp.second * tpA.second * tpB.second;
                    if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tfp.second) + log(tjp.second) + log(tpA.second) + log(tpB.second) + log(ktpr_p_t.second) + log(modP[ppredictor][t_p_t]) > beams[t].front().first() ) {
                      StoreState ss( q_tdec1, f, j, tpA.first, tpB.first, aPretrm );
                      if( (1<t && t<lwSent.size() && ss.size()>0) || (t==1 && ss.size()==0) || (t==lwSent.size() && ss.size()==0) ) {
                        if( beams[t].size()>=BEAM_WIDTH ) { pop_heap( beams[t].begin(), beams[t].end(), [] (const BeamElement& a, const BeamElement& b) { return b<a; } ); beams[t].pop_back(); }
                        beams[t].emplace_back( lgpr_tdec1 + log(tfp.second) + log(tjp.second) + log(tpA.second) + log(tpB.second) + log(ktpr_p_t.second) + log(modP[ppredictor][t_p_t]), aPretrm, f, j, ss, &be_tdec1-&beams[t-1][0] );
                        if( VERBOSE>1 ) cout << "                    send (" << be_tdec1.fifth() << ") to (" << ss << ") with " << (lgpr_tdec1 + log(tfp.second) + log(tjp.second) + log(tpA.second) + log(tpB.second) + log(ktpr_p_t.second) + log(modP[ppredictor][t_p_t])) << endl;
                        push_heap( beams[t].begin(), beams[t].end(), [] (const BeamElement& a, const BeamElement& b) { return b<a; } );
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Sort beam...
      std::sort( beams[t].begin(), beams[t].end(), [] (const BeamElement& a, const BeamElement& b) { return b<a; } );

      // Write output...
      cerr << " (" << beams[t].size() << ")";
      if( VERBOSE ) cout << "BEAM (" << w_t << ")\n" << beams[t] << endl;
    }
    cerr << endl;
    if( VERBOSE ) cout << "MLS" << endl;

    auto& mls = beams.getMostLikelySequence();
    if( mls.size()>0 ) {
      auto ibe=next(mls.begin()); auto iw=lwSent.begin(); for( ; next(ibe)!=mls.end() && next(iw)!=lwSent.end(); ibe++, iw++ )
        cout << *iw << " " << ibe->second() << " " << next(ibe)->third() << " " << next(ibe)->fourth() << " " << next(ibe)->fifth() << endl;
    }
    if( mls.size()==0 ) cout << "FAIL FAIL 1 1 " << endl;
  }
}

