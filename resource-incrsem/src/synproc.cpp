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

#define SERIAL_IO

////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";

////////////////////////////////////////////////////////////////////////////////

typedef T P;
typedef T A;
typedef T B;

////////////////////////////////////////////////////////////////////////////////

/*
class HiddState : public DelimitedSext<psX,Delimited<double>,psSpace,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psSpace,Delimited<int>,psX> {
 public:
  HiddState ( )                                                               : DelimitedSext<psX,Delimited<double>,psSpace,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psSpace,Delimited<int>,psX>()            { }
  HiddState ( double d, const Sign& a, F f, J j, const StoreState& q, int i ) : DelimitedSext<psX,Delimited<double>,psSpace,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psSpace,Delimited<int>,psX>(d,a,f,j,q,i) { }
};

class Beam : public DelimitedVector<psX,HiddState,psLine,psX> {
 public:
  Beam ( )        : DelimitedVector<psX,HiddState,psLine,psX>() { }
  //Beam ( uint i ) : priority_queue<HiddState,DelimitedVector<psX,HiddState,psLine,psX>,greater<HiddState>>() { reserve(i); }
  //iterator<HiddState>& begin() { return c.begin(); }
  //iterator<HiddState>& end() { return c.end(); }
};
*/

class HiddState : public DelimitedQuad<psX,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psX> {
 public:
  HiddState ( )                                              : DelimitedQuad<psX,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psX>()        { }
  HiddState ( const Sign& a, F f, J j, const StoreState& q ) : DelimitedQuad<psX,Sign,psSpace,F,psSpace,J,psSpace,StoreState,psX>(a,f,j,q) { }
};
const HiddState beStableDummy; //equivalent to "beStableDummy = HiddState()"

/*
//typedef pair<double,const HiddState&> ProbBack;
class ProbBack : public pair<double, const HiddState&> {
  public :
    ProbBack ( )                                  : pair<double, const HiddState&> ( 0.0, beStableDummy ) { }
    ProbBack ( double d , const HiddState& be ) : pair<double, const HiddState&> ( d,   be            ) { }
};
*/

class Trellis : public vector<Beam<HiddState>> {
// private:
//  DelimitedList<psX,pair<HiddState,ProbBack>,psLine,psX> lbe;
 public:
  Trellis ( ) : vector<Beam<HiddState>>() { reserve(100); }
  Beam<HiddState>& operator[] ( uint i ) { if ( i==size() ) emplace_back(BEAM_WIDTH); return vector<Beam<HiddState>>::operator[](i); }
//  const DelimitedList<psX,pair<HiddState,ProbBack>,psLine,psX>& getMostLikelySequence ( ) {
  void setMostLikelySequence( DelimitedVector<psX,DelimitedTrip<psX,ObsWord,psSpace,HiddState,psSpace,double,psX>,psLine,psX>& mls ) {
    //mls.clear( );
    //mls.reserve( size() );
    auto last = *back().begin();  //pair<const HiddState,ProbBack>( back().begin()->second, back().begin()->first );
    int t = size()-1;
//    if( back().size()>0 ) for( uint t = beams.size()-1; t>=0; t-- ) 
    if( back().size()>0 ) {
      // Add chain from best at end...
      for( const auto* pbe = &last; t>0; pbe = &pbe->getBack() ) mls[--t].second() = pbe->getHidd();
      // Shift storestates bc of dateline issue...
      for( int t=0; t<size()-1; t++ ) mls[t-1].second().fourth() = mls[t].second().fourth();
      mls[size()-2].second().fourth() = StoreState();
    }
//    lbe.clear();  if( back().size()>0 ) lbe.push_front( pair<HiddState,ProbBack>( back().begin()->second, back().begin()->first ) );
//    if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( at(t).get(lbe.front().second.second) );
//    if( lbe.size()>0 ) lbe.emplace_back( HiddState(), ProbBack(0.0,HiddState()) );
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

int main ( int nArgs, const char* argv[] ) {

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
      else if ( 0==strncmp(argv[a],"-p",2) ) numThreads = atoi(argv[a]+2);
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

#ifdef SERIAL_IO
  // List of sentences, which are pairs of lists of words and lists of hidd states...
  list< pair< DelimitedList<psX,ObsWord,psSpace,psX>, DelimitedVector<psX,DelimitedTrip<psX,ObsWord,psSpace,HiddState,psSpace,double,psX>,psLine,psX> > > corpus;
  uint linenum = 0;
  // Read in list of sentences...
  while( cin.peek() != EOF ) {
    corpus.emplace_back();
    cin >> corpus.back().first >> "\n";
    cerr<<"I read a sentence with " << corpus.back().first.size() << " words." << endl;
  }
  // Pointers to 
  auto isentNextToProc = corpus.begin();
  auto isentNextToDump = corpus.begin();
#else
  list<DelimitedVector<psX,DelimitedTrip<psX,ObsWord,psSpace,HiddState,psSpace,double,psX>,psLine,psX>> MLSs;  // list of most-likely-sequence vectors
  uint           linenum = 1;   // line number read
#endif
  mutex          mutexMLSList;  // mutex for MLS list
  vector<thread> vtWorkers;     // vector of worker threads
  StoreState     ssLongFail( StoreState(), 1, 0, "FAIL", "FAIL", "FAIL" );

  if( OUTPUT_MEASURES ) cout << "word pos f j store totsurp" << endl;

  // For each line in stdin...
//  for( int linenum=1; cin && EOF!=cin.peek(); linenum++ ) {
  for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {

   // Each thread loop until end of input...
   while( true ) {

    Trellis                                beams;  // sequence of beams
    uint                                   t=0;    // time step

    // Create initial beam element...
    StoreState ssInit; ssInit.emplace_back(Sign("S"),Sign("."));    //Sign("S-lS^g_0"),Sign("."));
    beams[0].tryAdd( HiddState(Sign("."),0,1,ssInit), ProbBack<HiddState>() );
 
#ifdef SERIAL_IO
    decltype(corpus)::iterator isent;
    uint                       currline;
    { lock_guard<mutex> guard( mutexMLSList );
      if( isentNextToProc == corpus.end() ) break;
      isent = isentNextToProc++;
      currline = linenum++;
      // Report sentence...
      cerr << "Worker " << numt << " processing sentence " << currline << ": " << isent->first << " ..." << endl;
    }
    const auto& lwSent = isent->first;
    auto&       mls    = isent->second;
#else
    DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // input list

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
#endif

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
        double            lgpr_tdec1 = be_tdec1.getProb();          // prob of prev storestate
        const Sign&       p_tdec1    = be_tdec1.getHidd().first();  // prev P
        F                 f_tdec1    = be_tdec1.getHidd().second(); // prev F
        J                 j_tdec1    = be_tdec1.getHidd().third();  // prev J
        const StoreState& q_tdec1    = be_tdec1.getHidd().fourth(); // prev storestate

        if( VERBOSE>1 ) cout << "  from (" << be_tdec1.getHidd() << ")" << endl;

        // For each possible apex category label...
        APredictor apredictor = q_tdec1.calcApexTypeCondition( f_tdec1, j_tdec1, p_tdec1 );  // save apredictor for use in prob calc
        if( VERBOSE>1 ) cout << "    A " << apredictor << "..." << endl;
        for( auto& tpA : modA[apredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tpA.second) > beams[t].rbegin()->getProb() ) {

          if( VERBOSE>1 ) cout << "    A " << apredictor << " : " << tpA.first << " = " << tpA.second << endl;

          // For each possible brink category label...
          BPredictor bpredictor = q_tdec1.calcBrinkTypeCondition( f_tdec1, j_tdec1, tpA.first, p_tdec1 );  // bpredictor for prob calc
          if( VERBOSE>1 ) cout << "      B " << bpredictor << "..." << endl;
          for( auto& tpB : modB[bpredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tpA.second) + log(tpB.second) > beams[t].rbegin()->getProb() ) {

            if( VERBOSE>1 ) cout << "      B " << bpredictor << " : " << tpB.first << " = " << tpB.second << endl;

            StoreState ss( q_tdec1, f_tdec1, j_tdec1, tpA.first, tpB.first, p_tdec1 );
            // For each possible lemma (context + label + prob) for preterminal of current word...
            for( auto& ktpr_p_t : (             lexW.end() != lexW.find( w_t)) ? lexW[w_t] :
                                  ( BERKUNK and lexW.end() != lexW.find( unkWordBerk( w_t.getString().c_str() ) ) ) ? lexW[ unkWordBerk( w_t.getString().c_str() ) ] :
                                  (             lexW.end() != lexW.find( unkWord(     w_t.getString().c_str() ) ) ) ? lexW[ unkWord(     w_t.getString().c_str() ) ] :
                                  list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>() )
             if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(tpA.second) + log(tpB.second) + log(ktpr_p_t.second) > beams[t].rbegin()->getProb() ) {
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
                  for( auto& tjp : modJ[jpredictor] ) if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(tjp.second) > beams[t].rbegin()->getProb() ) {
                    const J& j = tjp.first;
                    if( VERBOSE>1 ) cout << "              J " << jpredictor << " : " << tjp.first << " = " << tjp.second << endl;

                    // Calculate probability and storestate and add to beam...
                    //double probJoin = tjp.second * tpA.second * tpB.second;
//                    { lock_guard<mutex> guard( mutexBeam );
                      if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(tjp.second) > beams[t].rbegin()->getProb() ) {
                        //StoreState ss( q_tdec1, f, j, tpA.first, tpB.first, aPretrm );
                        if( (t<lwSent.size() && ss.size()+f-j>0) || (t==lwSent.size() && ss.size()+f-j==0) ) {
////                          if( beams[t].size()>=BEAM_WIDTH ) { pop_heap( beams[t].begin(), beams[t].end(), [] (const HiddState& a, const HiddState& b) { return b<a; } ); beams[t].pop_back(); }
                          beams[t].tryAdd( HiddState( aPretrm, f, j, ss ), ProbBack<HiddState>( lgpr_tdec1 + log(probFork) + log(tjp.second), be_tdec1 ) );
                          if( VERBOSE>1 ) cout << "                send (" << be_tdec1.getHidd() << ") to (" << ss << ") with " << (lgpr_tdec1 + log(probFork) + log(tjp.second)) << endl;
////                          push_heap( beams[t].begin(), beams[t].end(), [] (const HiddState& a, const HiddState& b) { return b<a; } );
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
////      std::sort( beams[t].begin(), beams[t].end(), [] (const HiddState& a, const HiddState& b) { return b<a; } );

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
      for( auto& w : lwSent ) mls.emplace_back( w, HiddState( "FAIL", 1, 1, ssLongFail ), 0.0/0.0 );
      if( mls.size()>1 ) {
        mls.front().second() = HiddState( "FAIL", 1, 0, ssLongFail );
        mls.back( ).second() = HiddState( "FAIL", 0, 1, StoreState() );
      }
      else mls.front().second() = HiddState( "FAIL", 1, 1, StoreState() );
      // Obtain MLS for current line...
      beams.setMostLikelySequence( mls );
      // Update measures...
      for( uint t=1; t<beams.size(); t++ ) {
        double probPrevTot = 0.0;
        double probCurrTot = 0.0;
        for( auto& be_tdec1 : beams[t-1] )
          probPrevTot += exp(be_tdec1.getProb() - beams[t-1].begin()->getProb());
        for( auto& be_t : beams[t] )
          probCurrTot += exp(be_t.getProb()     - beams[t-1].begin()->getProb());
        mls[t-1].third() = log2(probPrevTot) - log2(probCurrTot);
      }
#ifdef SERIAL_IO
      for( ; isentNextToDump != corpus.end() and isentNextToDump->first.size() == isentNextToDump->second.size(); isentNextToDump++ ) {
        cout << isentNextToDump->second << endl;
      }
#else
      // Dump all consecutive lines that are finished...
      while( MLSs.size()>0 && MLSs.front().size()>0 ) { cout << MLSs.front() << endl; MLSs.pop_front(); }
#endif
    }

//    auto& mls = beams.getMostLikelySequence();
//    if( mls.size()>0 ) {
//      int u=1; auto ibe=next(mls.begin()); auto iw=lwSent.begin(); for( ; ibe!=mls.end() && iw!=lwSent.end(); ibe++, iw++, u++ ) {
//        cout << *iw << " " << ibe->getProb()() << " " << ibe->first.second() << " " << ibe->first.third() << " " << next(ibe)->first.fourth();
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

