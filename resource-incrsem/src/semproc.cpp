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
#include <Beam.hpp>

uint BEAM_WIDTH = 1000;
uint VERBOSE    = 0;

////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";
char psSpaceF[]       = " f";
char psAmpersand[]    = "&";

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<T> P;
typedef Delimited<T> A;
typedef Delimited<T> B;

////////////////////////////////////////////////////////////////////////////////

class BeamElement : public DelimitedSext<psX,Sign,psSpaceF,F,psAmpersand,E,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psX> {
 public:
  BeamElement ( )                                                                 : DelimitedSext<psX,Sign,psSpaceF,F,psAmpersand,E,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psX>()             { }
  BeamElement ( const Sign& a, F f, E e, K k, JResponse jr, const StoreState& q ) : DelimitedSext<psX,Sign,psSpaceF,F,psAmpersand,E,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psX>(a,f,e,k,jr,q) { }
};

typedef pair<double,const BeamElement&> ProbBack;

class Trellis : public vector<Beam<ProbBack,BeamElement>> {
 private:
  DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX> lbe;
 public:
  Trellis ( ) : vector<Beam<ProbBack,BeamElement>>() { reserve(100); }
  Beam<ProbBack,BeamElement>& operator[] ( uint i ) { if ( i==size() ) emplace_back(BEAM_WIDTH); return vector<Beam<ProbBack,BeamElement>>::operator[](i); }
  const DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX>& getMostLikelySequence ( ) {
    lbe.clear();  if( back().size()>0 ) lbe.push_front( pair<BeamElement,ProbBack>( back().begin()->second, back().begin()->first ) );
    if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( at(t).get(lbe.front().second.second) );
    if( lbe.size()>0 ) lbe.emplace_back( BeamElement(), ProbBack(0.0,BeamElement()) );
    return lbe;
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

  uint numThreads = 1;

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
      if      ( 0==strcmp(argv[a],"-v") ) VERBOSE = 1;
      else if ( 0==strcmp(argv[a],"-V") ) VERBOSE = 2;
      else if ( 0==strncmp(argv[a],"-p",2) ) numThreads = atoi(argv[a]+2);
      else if ( 0==strncmp(argv[a],"-b",2) ) BEAM_WIDTH = atoi(argv[a]+2);
      else if ( 0==strncmp(argv[a],"-f",2) ) FEATCONFIG = atoi(argv[a]+2);
      //else if ( string(argv[a]) == "t" ) STORESTATE_TYPE = true;
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
    matF = arma::zeros( FResponse::getDomain().getSize(), FPredictor::getDomainSize() );
    matJ = arma::zeros( JResponse::getDomain().getSize(), JPredictor::getDomainSize() );
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

  // For each line in stdin...
  for ( int linenum=1; cin && EOF!=cin.peek(); linenum++ ) {

    Trellis                                beams;  // sequence of beams
    uint                                   t=0;    // time step
    DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // input list

    // Allocate space in beams to avoid reallocation...
    // Create initial beam element...
    beams[0].tryAdd( BeamElement(), ProbBack(0.0,BeamElement()) );
    // Read sentence...
    cin >> lwSent >> "\n";
    cerr << "#" << linenum;

    // For each word...
    for ( auto& w_t : lwSent ) {

      cerr << " " << w_t;
      if ( VERBOSE ) cout << "WORD:" << w_t << endl;

      // Create beam for current time step...
      beams[++t].clear();

      mutex mutexBeam;
      vector<thread> vtWorkers;
      for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {

        // For each hypothesized storestate at previous time step...
        uint i=0; for( auto& be_tdec1 : beams[t-1] ) if( i++%numThreads==numt ){
          double            lgpr_tdec1 = be_tdec1.first.first;      // prob of prev storestate
          const StoreState& q_tdec1    = be_tdec1.second.sixth();  // prev storestate

          if( VERBOSE>1 ) cout << "  from (" << be_tdec1.second << ")" << endl;

          // Calc distrib over response for each fork predictor...
          arma::vec fresponses = arma::zeros( matF.n_rows );
          list<FPredictor> lfpredictors;  q_tdec1.calcForkPredictors( lfpredictors, false );  lfpredictors.emplace_back();  // add bias term
          for ( auto& fpredr : lfpredictors ) if ( fpredr.toInt() < matF.n_cols ) fresponses += matF.col( fpredr.toInt() );
          if ( VERBOSE>1 ) for ( auto& fpredr : lfpredictors ) cout<<"    fpredr:"<<fpredr<<endl;
          fresponses = arma::exp( fresponses );

          // Calc normalization term over responses...
          double fnorm = arma::accu( fresponses );

          // For each possible lemma (context + label + prob) for preterminal of current word...
          for ( auto& ktpr_p_t : (lexW.end()!=lexW.find(w_t)) ? lexW[w_t] : lexW[unkWord(w_t.getString().c_str())] ) {
            if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(ktpr_p_t.second) > beams[t].rbegin()->first.first ) {
              K k_p_t           = (FEATCONFIG & 2) ? K::kBot : ktpr_p_t.first.first;   // context of current preterminal
              T t_p_t           = ktpr_p_t.first.second;                               // label of cunnent preterminal
              E e_p_t           = (t_p_t.getLastNonlocal()==N_NONE) ? 'N' : (t_p_t.getLastNonlocal()==N("-rN")) ? '0' : (t_p_t.getLastNonlocal().isArg()) ? t_p_t.getArity()+'1' : 'M';
              double probwgivkl = ktpr_p_t.second;                                     // probability of current word given current preterminal

              if ( VERBOSE>1 ) cout << "     W " << k_p_t << " " << t_p_t << " : " << w_t << " = " << probwgivkl << endl;

              // For each possible no-fork or fork decision...
              for ( auto& f : {0,1} ) {
                double scoreFork = ( FResponse::exists(f,e_p_t,k_p_t) ) ? fresponses(FResponse(f,e_p_t,k_p_t).toInt()) : 1.0 ;
                if ( VERBOSE>1 ) cout << "      F ... : " << f << " " << e_p_t << " " << k_p_t << " = " << (scoreFork / fnorm) << endl;

                // If preterminal prob is nonzero... 
                PPredictor ppredictor = q_tdec1.calcPretrmTypeCondition(f,e_p_t,k_p_t);
                if ( VERBOSE>1 ) cout << "      P " << ppredictor << "..." << endl;
                if ( modP.end()!=modP.find(ppredictor) && modP[ppredictor].end()!=modP[ppredictor].find(t_p_t) ) {

                  if ( VERBOSE>1 ) cout << "      P " << ppredictor << " : " << t_p_t << " = " << modP[ppredictor][t_p_t] << endl;

                  // Calc probability for fork phase...
                  double probFork = (scoreFork / fnorm) * modP[ppredictor][t_p_t] * probwgivkl;
                  if ( VERBOSE>1 ) cout << "      f: " << f <<"&"<< k_p_t << " " << scoreFork << " / " << fnorm << " * " << modP[ppredictor][t_p_t] << " * " << probwgivkl << " = " << probFork << endl;

                  Sign aPretrm;  aPretrm.first().emplace_back(k_p_t);  aPretrm.second() = t_p_t;  aPretrm.third() = S_A;          // aPretrm (pos tag)
                  const LeftChildSign aLchild( q_tdec1, f, e_p_t, aPretrm );
                  list<JPredictor> ljpredictors; q_tdec1.calcJoinPredictors( ljpredictors, f, e_p_t, aLchild, false ); // predictors for join
                  ljpredictors.emplace_back();                                                                  // add bias
                  arma::vec jresponses = arma::zeros( matJ.n_rows );
                  for ( auto& jpredr : ljpredictors ) if ( jpredr.toInt() < matJ.n_cols ) jresponses += matJ.col( jpredr.toInt() );
                  jresponses = arma::exp( jresponses );
                  double jnorm = arma::accu( jresponses );  // 0.0;                                           // join normalization term (denominator)

                  // For each possible no-join or join decision, and operator decisions...
                  for( JResponse jresponse; jresponse<JResponse::getDomain().getSize(); ++jresponse ) {
                    if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(jresponses[jresponse.toInt()]/jnorm) > beams[t].rbegin()->first.first ) {
                      J j   = jresponse.getJoin();
                      E e   = jresponse.getE();
                      O opL = jresponse.getLOp();
                      O opR = jresponse.getROp();
                      double probJoin = jresponses[jresponse.toInt()] / jnorm;
                      if ( VERBOSE>1 ) cout << "       J ... " << " : " << jresponse << " = " << probJoin << endl;

                      // For each possible apex category label...
                      APredictor apredictor = q_tdec1.calcApexTypeCondition( f, j, e_p_t, e, opL, aLchild );  // save apredictor for use in prob calc
                      if ( VERBOSE>1 ) cout << "         A " << apredictor << "..." << endl;
                      if ( modA.end()!=modA.find(apredictor) )
                       for ( auto& tpA : modA[apredictor] ) {
                        if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) > beams[t].rbegin()->first.first ) {

                          if ( VERBOSE>1 ) cout << "         A " << apredictor << " : " << tpA.first << " = " << tpA.second << endl;

                          // For each possible brink category label...
                          BPredictor bpredictor = q_tdec1.calcBrinkTypeCondition( f, j, e_p_t, e, opL, opR, tpA.first, aLchild );  // bpredictor for prob calc
                          if ( VERBOSE>1 ) cout << "          B " << bpredictor << "..." << endl;
                          if ( modB.end()!=modB.find(bpredictor) )
                           for ( auto& tpB : modB[bpredictor] ) {
                            if ( VERBOSE>1 ) cout << "          B " << bpredictor << " : " << tpB.first << " = " << tpB.second << endl;
                            lock_guard<mutex> guard( mutexBeam );
                            if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second) > beams[t].rbegin()->first.first ) {

                              // Calculate probability and storestate and add to beam...
                              StoreState ss( q_tdec1, f, j, e_p_t, e, opL, opR, tpA.first, tpB.first, aPretrm, aLchild );
                              if( (t<lwSent.size() && ss.size()>0) || (t==lwSent.size() && ss.size()==0) ) {
                                beams[t].tryAdd( BeamElement( aPretrm, f,e_p_t,k_p_t, jresponse, ss ), ProbBack( lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second), be_tdec1.second ) );
                                if( VERBOSE>1 ) cout << "                send (" << be_tdec1.second << ") to (" << ss << ") with "
                                                     << (lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second)) << endl;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }, numtglobal ));

      for( auto& w : vtWorkers ) w.join();

      // Write output...
      cerr << " (" << beams[t].size() << ")";
      if ( VERBOSE ) cout << beams[t] << endl;
    }
    cerr << endl;
    if ( VERBOSE ) cout << "MLS" << endl;

    auto& mls = beams.getMostLikelySequence();
    if( mls.size()>0 ) {
      int u=1; auto ibe=next(mls.begin()); auto iw=lwSent.begin(); for( ; ibe!=mls.end() && iw!=lwSent.end(); ibe++, iw++, u++ ) {
        cout << *iw << " " << ibe->first;
//ibe->first.first() << " " << ibe->first.second() << " " << ibe->first.third() << " " << ibe->first.fourth();
//        if( OUTPUT_MEASURES ) cout << " " << vdSurp[u];
        cout << endl;
      }
    }
    if( mls.size()==0 ) {
      cout << "FAIL FAIL 1 1 " << endl;
//      int u=1; auto iw=lwSent.begin(); for( ; iw!=lwSent.end(); iw++, u++ ) {
//        cout << *iw << " FAIL 1 1 ";
//        if( OUTPUT_MEASURES ) cout << " " << vdSurp[u];
//        cout << endl;
//      }
    }
  }
}

