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
#include <sstream>
#include <list>
#include <thread>
#include <mutex>
#include <chrono>
#include <regex>
#include <cmath>
using namespace std;
#include <armadillo>
using namespace arma;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
#include <Beam.hpp>
bool STORESTATE_TYPE = true;
bool STORESTATE_CHATTY = false;
uint FEATCONFIG = 0;
#include <StoreState.hpp>
#include <transformer.hpp>
#include <NModel_mlp.hpp>
#include <FModel_transformer.hpp>
#include <WModel_mlp.hpp>
#include <JModel_transformer.hpp>
#include <Trellis.hpp>

int COREF_WINDOW = INT_MAX;
bool ABLATE_UNARY = false;
bool NO_ENTITY_BLOCKING = false;
bool NO_ANTUNK = false;
bool REDUCED_PRTRM_CONTEXTS = false;
bool ORACLE_COREF = false;
#define SERIAL_IO

////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";
const W wEmpty = W("");

////////////////////////////////////////////////////////////////////////////////

uint VERBOSE         = 0;
uint OUTPUT_MEASURES = true;

////////////////////////////////////////////////////////////////////////////////


string correctString(string s) {
  if ( s == "[Top]" ) return "Top";
  if ( s == "[Bot]" ) return "Bot";
  return s;
}


vector<string> getTokStrings() {
//DelimitedVector<psX,ObsWord,psSpace,psX> getToks() {
  DelimitedList<psX,ObsWord,psSpace,psX> toks;
  //DelimitedVector<psX,ObsWord,psSpace,psX> toks;
  ostringstream oss;
  cin >> toks >> "\n";
  oss << toks << endl;
  istringstream iss = istringstream(oss.str());
  vector<string> tokens{istream_iterator<string>{iss},
                        istream_iterator<string>{}};
  return tokens;
}

DelimitedVector<psX,ObsWord,psSpace,psX> getTokWs() {
  DelimitedVector<psX,ObsWord,psSpace,psX> toks;
  cin >> toks >> "\n";
  return toks;
}

int main ( int nArgs, char* argv[] ) {


  string sentDelim = "!SENTENCE";

  // Define model structures...
  EMat                          matEmutable;
  OFunc                         funcOmutable;
  NModel                        modNmutable;
  FModel                        modFmutable;
  PModel                        modPmutable;
  WModel                        modWmutable;
//  map<W,list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>> lexW;
  JModel                        modJmutable;
  AModel                        modAmutable;
  BModel                        modBmutable;
  map<int,map<int,map<int,int>>> oracle_dswo;

  list<DelimitedTrip<psX,WPredictor,psSpcColonSpc,W,psSpcEqualsSpc,Delimited<double>,psX>> lW;

  cerr << "Loading model " << argv[1] << "..." << endl;
  // Open file...
  ifstream fin (argv[1], ios::in );
  // Read model lists...
  int linenum = 0;
  while ( fin && EOF!=fin.peek() ) {
    if      ( fin.peek()=='E' ) matEmutable = EMat( fin );
    else if ( fin.peek()=='O' ) funcOmutable = OFunc( fin );
    else if ( fin.peek()=='N' ) { modNmutable = NModel( fin ); cerr << "loaded N model" << endl; }
    else if ( fin.peek()=='F' ) { modFmutable = FModel( fin ); cerr << "loaded F model" << endl; }
    else if ( fin.peek()=='P' ) { modPmutable = PModel( fin ); cerr << "loaded P model" << endl; }
    else if ( fin.peek()=='W' ) { modWmutable = WModel( fin ); cerr << "loaded W model" << endl; } 
    else if ( fin.peek()=='J' ) { modJmutable = JModel( fin ); cerr << "loaded J model" << endl; }
    else if ( fin.peek()=='A' ) { modAmutable = AModel( fin ); cerr << "loaded A model" << endl; }
    else if ( fin.peek()=='B' ) { modBmutable = BModel( fin ); cerr << "loaded B model" << endl; }
    else {
      Delimited<string> sOffSpec;
      fin >> sOffSpec >> "\n";
      cerr << "WARNING: skipping off-spec input line: '" << sOffSpec << "'" << endl;
    } 
    if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
  }
  cerr << "Model " << argv[1] << " loaded." << endl;

  const EMat&   matE  = matEmutable;
  const OFunc&  funcO = funcOmutable;
//  const NModel& modN  = modNmutable;
  const FModel& modF  = modFmutable;
//  const PModel& modP  = modPmutable;
  const WModel& modW  = modWmutable;
  const JModel& modJ  = modJmutable;
  const AModel& modA  = modAmutable;
  const BModel& modB  = modBmutable;

  cerr<<"Models ready."<<endl;

  if ( cin.peek() != '!' ) throw "Wrong input format.";
  cin >> sentDelim >> "\n";

  vector<string> tokStrs;
  DelimitedVector<psX,ObsWord,psSpace,psX> tokWs;
  string fek_str;
  W w_t;
  string jeoo_str;
  string a_cvar_str;
  string b_cvar_str;
  uint word_index = 1;

  Trellis   beams;
  uint      t=0;

  // create an initial null beam element
  DelimitedList<psX,BeamElement<HiddState>,psLine,psX> dummyLbe;
  dummyLbe.emplace_back(); //create null beamElement
  dummyLbe.back().setProb() = 0.0;
  beams[0].tryAdd( dummyLbe.back().getHidd(), dummyLbe.back().getProbBack() );


  while ( cin.peek() != EOF ) {
    // create beam for current time step
    beams[++t].clear();

    // reset when a new sentence is reached
//    if ( cin.peek() == '!' ) {
//      cin >> sentDelim >> "\n";
//      pbe_curr = bels.begin();
//      word_index = 1;
//    }


    WModel::WWPPMap mapWWPP;
    WModel::XPMap mapXP;
    WModel::MPMap mapMP;
    WModel::WPPMap mapWPP;

    ////////////////
    // Collect info about the F, W, J, A, and B decisions at this time step
    ////////////////

    if ( cin.peek() != 'F' ) throw "Wrong input format.";
    tokStrs = getTokStrings();
    fek_str = correctString(tokStrs[8]);

    if ( cin.peek() != 'W' ) throw "Wrong input format.";
    tokWs = getTokWs();
    w_t = tokWs[7];
    cout << "\n ==== word: " << w_t << " ==== " << endl;

    if ( cin.peek() != 'J' ) throw "Wrong input format.";
    tokStrs = getTokStrings();
    jeoo_str = correctString(tokStrs[8]);

    if ( cin.peek() != 'A' ) throw "Wrong input format.";
    tokStrs = getTokStrings();
    a_cvar_str = correctString(tokStrs[9]);

    if ( cin.peek() != 'B' ) throw "Wrong input format.";
    tokStrs = getTokStrings();
    b_cvar_str = correctString(tokStrs[10]);

    if ( beams[t-1].size() != 1 ) throw "Multiple beam elements in beam.";
    for( const BeamElement<HiddState>& be_tdec1 : beams[t-1] ) { 
      const StoreState& q_tdec1 = be_tdec1.getHidd().sixth();  // prev storestate

      ////////////////
      // F decision
      ////////////////

      FPredictorVec lfpredictors( be_tdec1, hvTop, 0 );
      arma::vec fresponses = modF.calcResponses( lfpredictors, word_index-1 );

      modW.calcPredictorLikelihoods(w_t, mapWWPP, mapXP, mapMP, mapWPP);
      mapWWPP.try_emplace(w_t, mapWPP);

      bool fek_found = false;
      uint f_t;
      EVar e_p_t;
      K k_p_t;
      CVar c_p_t;
      for ( auto& ektpr_p_t : mapWPP ) { //ektpr_p_t is a pair of (Wpredictor, prob)
        e_p_t = ektpr_p_t.first.first();
        k_p_t = ektpr_p_t.first.second(); // context of current preterminal
        c_p_t = ektpr_p_t.first.third();  // label of current preterminal

        // For each possible no-fork or fork decision...
        for ( auto& f : {0,1} ) {
          ostringstream oss_curr;
          oss_curr << f << "&" << e_p_t << "&" << k_p_t;
          string fek_str_curr = oss_curr.str();
          if ( fek_str_curr != fek_str ) continue;
          else {
            f_t = f;
            fek_found = true;
            // break out of the outer for loop
            goto verify_fek_found;
          }
        }
      }

      verify_fek_found:
      if ( !fek_found ) throw "FEK not found";
      double fek_prob = fresponses( modF.getResponseIndex(f_t, e_p_t, k_p_t) );
      cout << "FEK: " << fek_str << " prob: " << fek_prob << endl;

      ////////////////
      // J decision
      ////////////////

      StoreState qPretrm( q_tdec1, f_t, true, hvTop, e_p_t, k_p_t, c_p_t, matE, funcO );
      const Sign& aPretrm = qPretrm.getApex();
      StoreState qTermPhase( qPretrm, f_t );
      const Sign& aLchild = qTermPhase.getApex();

      JPredictorVec ljpredictors( be_tdec1, f_t, aLchild );
      arma::vec jresponses = modJ.calcResponses( ljpredictors, word_index-1, 1 );

      bool jeoo_found = false;
      J j;
      EVar e;
      O opL;
      O opR;
      uint jresponse;
      for( jresponse=0; jresponse<jresponses.size(); jresponse++ ) {  //JResponse jresponse; jresponse<JResponse::getDomain().getSize(); ++jresponse )
        j   = modJ.getJEOO( jresponse ).first();
        e   = modJ.getJEOO( jresponse ).second();
        opL = modJ.getJEOO( jresponse ).third();
        opR = modJ.getJEOO( jresponse ).fourth();

        ostringstream oss_curr;
        oss_curr << j << "&" << e << "&" << opL << "&" << opR;
        string jeoo_str_curr = oss_curr.str();
        cout << "curr string: " << jeoo_str_curr << endl;
        if ( jeoo_str_curr == jeoo_str ) {
          cout << "j found!" << endl;
          jeoo_found = true;
          break;
        }
      }

      if ( !jeoo_found ) throw "JEOO not found";
      double jeoo_prob = jresponses[jresponse]; //  / jnorm;
      cout << "JEOO: " << jeoo_str << " prob: " << jeoo_prob << endl;

      ////////////////
      // A decision
      ////////////////

      bool a_found = false;
      CVar a_cvar;
      for ( auto& apred2map : modA ) {
        for ( auto& cvar2double : apred2map.second ) {
          a_cvar = cvar2double.first;
          ostringstream oss_curr;
          oss_curr << a_cvar;
          string a_cvar_str_curr = oss_curr.str();
          if ( a_cvar_str_curr == a_cvar_str ) {
            a_found = true;
            goto verify_a_found;
          }
        }
      }

      verify_a_found:
      if ( !a_found) throw "A not found";


      ////////////////
      // B decision
      ////////////////

      bool b_found = false;
      CVar b_cvar;
      for ( auto& bpred2map : modB ) {
        for ( auto& cvar2double : bpred2map.second ) {
          b_cvar = cvar2double.first;
          ostringstream oss_curr;
          oss_curr << b_cvar;
          string b_cvar_str_curr = oss_curr.str();
          if ( b_cvar_str_curr == b_cvar_str ) {
            b_found = true;
            goto verify_b_found;
          }
        }
      }

      verify_b_found:
      if ( !b_found ) throw "B not found";

      ////////////////
      // Update store state and beam element
      ////////////////
      StoreState ss( qTermPhase, j, e, opL, opR, a_cvar, b_cvar );
      beams[t].tryAdd( HiddState( aPretrm, f_t ,e_p_t, k_p_t, jresponse, ss, 0, w_t ), ProbBack<HiddState>( 0.0, be_tdec1 ) );
    }
  }
} 
