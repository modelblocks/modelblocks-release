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
using namespace std;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
bool STORESTATE_TYPE = true;
#include <RandoUnkWord.hpp>
#include <BerkUnkWord.hpp>
#include <StoreStateSynProc.hpp>
#include <Beam.hpp>

uint BEAM_WIDTH      = 1000;
uint VERBOSE         = 0;
uint OUTPUT_MEASURES = 0;
bool BERKUNK         = false;

////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";

////////////////////////////////////////////////////////////////////////////////

typedef T P;
typedef T A;
typedef T B;

////////////////////////////////////////////////////////////////////////////////

/*
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
*/

class BeamElement : public DelimitedQuad<psX,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psX> {
 public:
  BeamElement ( )                                              : DelimitedQuad<psX,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psX>()        { }
  BeamElement ( const Sign& a, F f, J j, const StoreState& q ) : DelimitedQuad<psX,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psX>(a,f,j,q) { }
};
const BeamElement beStableDummy; //equivalent to "beStableDummy = BeamElement()"

//typedef pair<double,const BeamElement&> ProbBack;
class ProbBack : public pair<double, const BeamElement&> {
  public :
    ProbBack ( )                                  : pair<double, const BeamElement&> ( 0.0, beStableDummy ) { }
    ProbBack ( double d , const BeamElement& be ) : pair<double, const BeamElement&> ( d,   be            ) { }
};

class Trellis : public vector<Beam<ProbBack,BeamElement>> {
// private:
//  DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX> lbe;
 public:
  Trellis ( ) : vector<Beam<ProbBack,BeamElement>>() { reserve(100); }
  Beam<ProbBack,BeamElement>& operator[] ( uint i ) { if ( i==size() ) emplace_back(BEAM_WIDTH); return vector<Beam<ProbBack,BeamElement>>::operator[](i); }
//  const DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX>& getMostLikelySequence ( ) {
  void setMostLikelySequence( DelimitedVector<psX,DelimitedTrip<psX,ObsWord,psSpace,BeamElement,psSpace,double,psX>,psLine,psX>& mls ) {
    //mls.clear( );
    //mls.reserve( size() );
    auto last = pair<const BeamElement,ProbBack>( back().begin()->second, back().begin()->first );
    int t = size()-1;
//    if( back().size()>0 ) for( uint t = beams.size()-1; t>=0; t-- ) 
    if( back().size()>0 ) {
      // Add chain from best at end...
      for( const auto* pbbe = &last; t>0; pbbe = &at(--t).get(pbbe->second.second) ) mls[t-1].second() = pbbe->first;
      // Shift storestates bc of dateline issue...
      for( int t=0; t<size()-1; t++ ) mls[t-1].second().fourth() = mls[t].second().fourth();
      mls[size()-2].second().fourth() = StoreState();
    }
//    lbe.clear();  if( back().size()>0 ) lbe.push_front( pair<BeamElement,ProbBack>( back().begin()->second, back().begin()->first ) );
//    if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( at(t).get(lbe.front().second.second) );
//    if( lbe.size()>0 ) lbe.emplace_back( BeamElement(), ProbBack(0.0,BeamElement()) );
//    return lbe;
  }
};

/*
class StreamTrellis : public vector<Beam> {
 public:
  StreamTrellis ( ) : vector<Beam>(2) { }       // previous and next beam.
  Beam&       operator[] ( uint i )       { return vector<Beam>::operator[](i%2); }
  const Beam& operator[] ( uint i ) const { return vector<Beam>::operator[](i%2); }
};
*/

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  uint numThreads = 10;
  bool bUnkShrink = false;

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

    lA.emplace_back( DelimitedTrip<psX,APredictor,psSpcColonSpc,A,psSpcEqualsSpc,Delimited<double>,psX>(APredictor(0,1,T("T"),T("-")),A("-"),1.0) );      // should be T("S")
    lB.emplace_back( DelimitedTrip<psX,BPredictor,psSpcColonSpc,B,psSpcEqualsSpc,Delimited<double>,psX>(BPredictor(0,1,T("-"),T("S")),B("T"),1.0) );

    // For each command-line flag or model file...
    for( int a=1; a<nArgs; a++ ) {
      if      ( 0==strcmp(argv[a],"-v") ) VERBOSE = 1;
      else if ( 0==strcmp(argv[a],"-V") ) VERBOSE = 2;
      else if ( 0==strcmp(argv[a],"-u") ) BERKUNK = true;
      else if ( 0==strncmp(argv[a],"-b",2) ) BEAM_WIDTH = atoi(argv[a]+2);
      else if ( 0==strcmp(argv[a],"-c") ) OUTPUT_MEASURES = 1;
      else if ( 0==strcmp(argv[a],"-d") ) NODEP = true;
      else if ( 0==strcmp(argv[a],"-m") ) bUnkShrink = true;
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

  // Add unk...
  for( auto& entry : lexW ){
    // for each word:{<category:prob>}
    for( auto& unklistelem : lexW[ (BERKUNK) ? unkWordBerk(entry.first.getString().c_str()) : unkWord(entry.first.getString().c_str()) ] ){
      // for each possible unked(word) category:prob pair
      bool BAIL = false;
      for( auto& listelem : entry.second ) {
        if (listelem.first == unklistelem.first) {
          BAIL = true;
          listelem.second = listelem.second + ( ((bUnkShrink)?0.000001:1.0) * unklistelem.second ); // merge actual likelihood and unk likelihood
        }
      }
      if (not BAIL) entry.second.push_back(unklistelem);
    }
  }

  cerr<<"Models ready."<<endl;

  list<DelimitedVector<psX,DelimitedTrip<psX,ObsWord,psSpace,BeamElement,psSpace,double,psX>,psLine,psX>> MLSs;  // list of most-likely-sequence vectors
  mutex          mutexMLSList;  // mutex for MLS list
  vector<thread> vtWorkers;     // vector of worker threads
  uint           linenum = 1;   // line number read
  StoreState     ssLongFail( StoreState(), 1, 0, "FAIL", "FAIL", "FAIL" );

  if( OUTPUT_MEASURES ) cout << "word pos f j store totsurp" << endl;

  // For each line in stdin...
//  for( int linenum=1; cin && EOF!=cin.peek(); linenum++ ) {
  for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {

   // Each thread loop until end of input...
   while( true ) {

    Trellis                                beams;  // sequence of beams
    uint                                   t=0;    // time step
    DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // input list

    // Create initial beam element...
    StoreState ssInit; ssInit.emplace_back(Sign("S"),Sign("."));    //Sign("S-lS^g_0"),Sign("."));
    beams[0].tryAdd( BeamElement(Sign("."),0,1,ssInit), ProbBack(0.0,BeamElement()) );
 
    // Lock...
    mutexMLSList.lock( );
    // Unlock and bail if end of input...
    if( not ( cin && EOF!=cin.peek() ) ) { mutexMLSList.unlock(); break; }
    // Read sentence...
    uint currline = linenum++;
    cin >> lwSent >> "\n";
    // Report sentence...
    cerr << "Reading sentence " << currline << ": " << lwSent << " ..." << endl;
    // Add MLS to list...
    MLSs.emplace_back( );
    // Retain ref to MLS of current thread...
    auto& mls = MLSs.back( );
    // Unlock...
    mutexMLSList.unlock( );

//    vector<double> vdSurp(lwSent.size()+1); // vector for surprisal at each time step

    // For each word...
    for( auto& w_t : lwSent ) {

      if( numThreads == 1 ) cerr << " " << w_t;
      if( VERBOSE ) cout << "WORD:" << w_t << endl;

      // Create beam for current time step...
      beams[++t].clear();

//      mutex mutexBeam;
//      vector<thread> vtWorkers;
//      for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {

      // For each hypothesized storestate at previous time step...
      uint i=0; for( auto& be_tdec1 : beams[t-1] ) {
	// if( i++%numThreads==numt ){
        //for( auto& i=beams[t-1].begin()+numt; i<beams[t-1].end(); i+=numThreads ) {
        // auto&             be_tdec1   = *i; //beams[t-1][i];
        double            lgpr_tdec1 = be_tdec1.first.first;     // prob of prev storestate
        const Sign&       p_tdec1    = be_tdec1.second.first();  // prev P
        F                 f_tdec1    = be_tdec1.second.second(); // prev F
        J                 j_tdec1    = be_tdec1.second.third();  // prev J
        const StoreState& q_tdec1    = be_tdec1.second.fourth(); // prev storestate

        if( VERBOSE>1 ) cout << "  from (" << be_tdec1.second << ")" << endl;

        // For each possible apex category label...
        APredictor apredictor = q_tdec1.calcApexTypeCondition( f_tdec1, j_tdec1, p_tdec1 );  // save apredictor for use in prob calc
        if( VERBOSE>1 ) cout << "    A " << apredictor << "..." << endl;
        for( auto& tpA : modA[apredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tpA.second) > beams[t].rbegin()->first.first ) {

          if( VERBOSE>1 ) cout << "    A " << apredictor << " : " << tpA.first << " = " << tpA.second << endl;

          // For each possible brink category label...
          BPredictor bpredictor = q_tdec1.calcBrinkTypeCondition( f_tdec1, j_tdec1, tpA.first, p_tdec1 );  // bpredictor for prob calc
          if( VERBOSE>1 ) cout << "      B " << bpredictor << "..." << endl;
          for( auto& tpB : modB[bpredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tpA.second) + log(tpB.second) > beams[t].rbegin()->first.first ) {

            if( VERBOSE>1 ) cout << "      B " << bpredictor << " : " << tpB.first << " = " << tpB.second << endl;

            StoreState ss( q_tdec1, f_tdec1, j_tdec1, tpA.first, tpB.first, p_tdec1 );
            // For each possible lemma (context + label + prob) for preterminal of current word...
            for( auto& ktpr_p_t : (lexW.end()!=lexW.find(w_t)) ? lexW[w_t] : lexW[ (BERKUNK) ? unkWordBerk(w_t.getString().c_str()) : unkWord(w_t.getString().c_str()) ] )
             if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tpA.second) + log(tpB.second) + log(ktpr_p_t.second) > beams[t].rbegin()->first.first ) {
              T t_p_t           = ktpr_p_t.first;  // label of current preterminal
              double probwgivkl = ktpr_p_t.second; // probability of current word given current preterminal

              if( VERBOSE>1 ) cout << "        W " << t_p_t << " : " << w_t << " = " << probwgivkl << endl;

              // For each possible no-fork or fork decision...
              //for ( auto& f : {0,1} ) {
              FPredictor fpredictor = ss.calcForkTypeCondition();
              for( auto& tfp : modF[fpredictor] ) {
                const F& f = tfp.first;
                if( VERBOSE>1 ) cout << "          F " << fpredictor << " : " << tfp.first << " = " << tfp.second << endl;

                // If preterminal prob is nonzero... 
                PPredictor ppredictor = ss.calcPretrmTypeCondition(f);
                if( VERBOSE>1 ) cout << "            P " << ppredictor << "..." << endl;
                if( modP[ppredictor].end()!=modP[ppredictor].find(t_p_t) ) {

                  if( VERBOSE>1 ) cout << "            P " << ppredictor << " : " << t_p_t << " = " << modP[ppredictor][t_p_t] << endl;

                  // Calc probability for fork phase...
                  double probFork = tpA.second * tpB.second * tfp.second * modP[ss.calcPretrmTypeCondition(f)][t_p_t] * probwgivkl;

                  //const Sign& aAncstr = q_tdec1.getAncstr( f );                                        // aAncstr (brink of previous incomplete cat
                  Sign aPretrm = t_p_t;                                                                // aPretrm (pos tag)
                  // For each possible no-join or join decision...
                  //for ( auto& j : {0,1} ) {
                  JPredictor jpredictor = ss.calcJoinTypeCondition(f,aPretrm);
                  for( auto& tjp : modJ[jpredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(tjp.second) > beams[t].rbegin()->first.first ) {
                    const J& j = tjp.first;
                    if( VERBOSE>1 ) cout << "              J " << jpredictor << " : " << tjp.first << " = " << tjp.second << endl;

                    // Calculate probability and storestate and add to beam...
                    //double probJoin = tjp.second * tpA.second * tpB.second;
//                    { lock_guard<mutex> guard( mutexBeam );
                      if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(tjp.second) > beams[t].rbegin()->first.first ) {
                        //StoreState ss( q_tdec1, f, j, tpA.first, tpB.first, aPretrm );
                        if( (t<lwSent.size() && ss.size()+f-j>0) || (t==lwSent.size() && ss.size()+f-j==0) ) {
////                          if( beams[t].size()>=BEAM_WIDTH ) { pop_heap( beams[t].begin(), beams[t].end(), [] (const BeamElement& a, const BeamElement& b) { return b<a; } ); beams[t].pop_back(); }
                          beams[t].tryAdd( BeamElement( aPretrm, f, j, ss ), ProbBack( lgpr_tdec1 + log(probFork) + log(tjp.second), be_tdec1.second ) );
                          if( VERBOSE>1 ) cout << "                send (" << be_tdec1.second << ") to (" << ss << ") with " << (lgpr_tdec1 + log(probFork) + log(tjp.second)) << endl;
////                          push_heap( beams[t].begin(), beams[t].end(), [] (const BeamElement& a, const BeamElement& b) { return b<a; } );
                        }
                      }
//                    }
                  }
                }
              }
            }
          }
        }
      }
//      }, numtglobal ));

//      for( auto& w : vtWorkers ) w.join();

////      // Sort beam...
////      std::sort( beams[t].begin(), beams[t].end(), [] (const BeamElement& a, const BeamElement& b) { return b<a; } );

//      // Write output...
//      cerr << " (" << beams[t].size() << ")";
//      if( VERBOSE ) cout << "BEAM (" << w_t << ")\n" << beams[t] << endl;
    }
//    cerr << endl;
//    if( VERBOSE ) cout << "MLS" << endl;

    { lock_guard<mutex> guard( mutexMLSList );
      // Report line finished...
      if( numtglobal > 1 ) cerr << "Finished line " << currline << " (" << beams[t].size() << ")..." << endl;
//cerr<<"beams size="<<beams.size()<<endl;
//cerr<<"lwSent size="<<lwSent.size()<<endl;
      // Add words and default fail elements to MLS...
      for( auto& w : lwSent ) mls.emplace_back( w, BeamElement( "FAIL", 1, 1, ssLongFail ), 0.0/0.0 );
      if( mls.size()>1 ) {
        mls.front().second() = BeamElement( "FAIL", 1, 0, ssLongFail );
        mls.back( ).second() = BeamElement( "FAIL", 0, 1, StoreState() );
      }
      else mls.front().second() = BeamElement( "FAIL", 1, 1, StoreState() );
      // Obtain MLS for current line...
      beams.setMostLikelySequence( mls );
      // Update measures...
      for( uint t=1; t<beams.size(); t++ ) {
        double probPrevTot = 0.0;
        double probCurrTot = 0.0;
        for( auto& be_tdec1 : beams[t-1] )
          probPrevTot += exp(be_tdec1.first.first - beams[t-1].begin()->first.first);
        for( auto& be_t : beams[t] )
          probCurrTot += exp(be_t.first.first     - beams[t-1].begin()->first.first);
        mls[t-1].third() = log2(probPrevTot) - log2(probCurrTot);
      }
      // Dump all consecuative lines that are finished...
      while( MLSs.size()>0 && MLSs.front().size()>0 ) { cout << MLSs.front() << endl; MLSs.pop_front(); }
    }

//    auto& mls = beams.getMostLikelySequence();
//    if( mls.size()>0 ) {
//      int u=1; auto ibe=next(mls.begin()); auto iw=lwSent.begin(); for( ; ibe!=mls.end() && iw!=lwSent.end(); ibe++, iw++, u++ ) {
//        cout << *iw << " " << ibe->first.first() << " " << ibe->first.second() << " " << ibe->first.third() << " " << next(ibe)->first.fourth();
//        if( OUTPUT_MEASURES ) cout << " " << vdSurp[u];
//        cout << endl;
//      }
//    }
//    if( mls.size()==0 ) {
//      uint u=1; auto iw=lwSent.begin(); for( ; iw!=lwSent.end(); iw++, u++ ) {
//        if( u==1 && lwSent.size()==1 ) cout << *iw << " FAIL 1 1 ";
//        else if( u==1 )                cout << *iw << " FAIL 1 0 FAIL/FAIL ";
//        else if( u<lwSent.size() )   cout << *iw << " FAIL 1 1 FAIL/FAIL ";
//        else                           cout << *iw << " FAIL 0 1 ";
//        if( OUTPUT_MEASURES ) cout << " " << vdSurp[u];
//        cout << endl;
//      }
//    }
        
    //if( mls.size()==0 ) cout << "FAIL FAIL 1 1 " << endl;
   }
  }, numtglobal ));

  for( auto& w : vtWorkers ) w.join();
}

