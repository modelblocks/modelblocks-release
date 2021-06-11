#define ARMA_64BIT_WORD
#include <iostream>
#include <fstream>
#include <list>
#include <thread>
#include <mutex>
#include <chrono>
#include <regex>
using namespace std;
#include <armadillo>
using namespace arma;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
#include <Beam.hpp>
uint FEATCONFIG = 0;
#include <StoreState.hpp>
#include <mlp.hpp>
#include <NModel_mlp.hpp>
#include <FModel_transformer.hpp>
#include <WModel_mlp.hpp>
// TODO switch to JModel_transformer.hpp
#include <JModel_mlp.hpp>

bool REDUCED_PRTRM_CONTEXTS = false;

int main ( int nArgs, char* argv[] ) {

  uint numThreads = 1;

  // Define model structures...
  FModel                        modFmutable;

  // For each command-line flag or model file...
  for ( int a=1; a<nArgs; a++ ) {
    cerr << "Loading model " << argv[a] << "..." << endl;
    // Open file...
    ifstream fin (argv[a], ios::in );
    modFmutable = FModel( fin ); cerr << "loaded F model" << endl;
  } //closes for int a=1

  const FModel& modF  = modFmutable;

 // vector<BeamElement<HiddState>> vbe;
 // vbe.emplace_back();
 //       //lbeWholeArticle.back().setProb() = 0.0;
 // vbe.back().setProb() = 0.0;
  //const HVec& hvAnt = HVec();
  BeamElement<HiddState> be0;
  be0.setProb() = 0.0;

  BeamElement<HiddState> be1 = BeamElement<HiddState>(be0.getProbBack(), be0.getHidd());
  
  const HVec& hvAnt = hvTop;
  const bool corefOn = 0;

  FPredictorVec fpv = FPredictorVec( be1, hvAnt, not corefOn );
  //FPredictorVec fpv = FPredictorVec( vbe.back(), hvAnt, not corefOn );
  vec fresponses = modF.calcResponses( fpv );
  //const StoreState& q_tdec1    = be_tdec1.getHidd().sixth();  // prev storestate
  //const HVec& hvAnt = pbeAnt->getHidd().getPrtrm().getHVec();
  //StoreState qPretrm( q_tdec1, f, REDUCED_PRTRM_CONTEXTS, hvAnt, e_p_t, k_p_t, c_p_t, matE, funcO );
  //StoreState qTermPhase( qPretrm, f );
  //StoreState ss( qTermPhase, j, e, opL, opR, cpA.first, cpB.first );
  //const Sign& aPretrm = qPretrm.getApex();
  //beams[t].tryAdd( HiddState( aPretrm, f,e_p_t,k_p_t, jresponse, ss, tAnt-t, w_t ), ProbBack<HiddState>( lgprItem, be_tdec1 ) );
  //FPredictorVec lfpredictors( be_tdec1, hvAnt, not corefON );

//DelimitedList<psX,BeamElement<HiddState>,psLine,psX> lbeWholeArticle;
//lbeWholeArticle.emplace_back(); //create null beamElement at start of article

}

