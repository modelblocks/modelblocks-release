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
#include <StoreStateCoref.hpp>
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
  BeamElement ( )                                                                 : DelimitedSext<psX,Sign,psSpaceF,F,psAmpersand,E,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psX>()             {third() = 'N';}
  BeamElement ( const Sign& a, F f, E e, K k, JResponse jr, const StoreState& q ) : DelimitedSext<psX,Sign,psSpaceF,F,psAmpersand,E,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psX>(a,f,e,k,jr,q) { }
};

// WS: I ADDED THIS
const BeamElement beStableDummy; //equivalent to "beStableDummy = BeamElement()"

//typedef pair<double,const BeamElement&> ProbBack;
class ProbBack : public pair<double, const BeamElement&> {
  public : 
    ProbBack ( ) : pair<double, const BeamElement&> ( 0.0, beStableDummy ) { }
    ProbBack (double d , const BeamElement& be) : pair<double, const BeamElement&> (d, be) {}
};

class Trellis : public vector<Beam<ProbBack,BeamElement>> {
  // private:
  //  DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX> lbe;
public:
  Trellis ( ) : vector<Beam<ProbBack,BeamElement>>() { reserve(100); }
  Beam<ProbBack,BeamElement>& operator[] ( uint i ) { if ( i==size() ) emplace_back(BEAM_WIDTH); return vector<Beam<ProbBack,BeamElement>>::operator[](i); }
  void setMostLikelySequence ( DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX>& lbe ) {
    lbe.clear();  if( back().size()>0 ) lbe.push_front( pair<BeamElement,ProbBack>( back().begin()->second, back().begin()->first ) );
    if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( at(t).get(lbe.front().second.second) );
    if( lbe.size()>0 ) lbe.emplace_back( BeamElement(), ProbBack(0.0,BeamElement()) );
    if( lbe.size()==0 ) {
      lbe.emplace_front( BeamElement( Sign(ksBot,"FAIL",0), 1, 'N', K::kBot, JResponse(1,'N','N','N'), StoreState() ), ProbBack(0.0,BeamElement()) );
      lbe.emplace_front( BeamElement( Sign(ksBot,"FAIL",0), 1, 'N', K::kBot, JResponse(1,'N','N','N'), StoreState() ), ProbBack(0.0,BeamElement()) );
      cerr<<"i failed"<<endl;
    }
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
    } //for a in nArgs

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

  list<DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX>> MLSs;
  list<DelimitedList<psX,ObsWord,psSpace,psX>> sents;
  mutex mutexMLSList;
  vector<thread> vtWorkers;
  uint linenum = 1;

  // For each line in stdin...
  //  for ( int linenum=1; cin && EOF!=cin.peek(); linenum++ ) {
  for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {

    while( true ) {

      Trellis                                beams;  // sequence of beams
      uint                                   t=0;    // time step
      DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // input list

      // Allocate space in beams to avoid reallocation...
      // Create initial beam element...
      //beams[0].tryAdd( BeamElement(), ProbBack(0.0, BeamElement()) );
      beams[0].tryAdd( BeamElement(), ProbBack());
      //cerr << "printing empty BeamElement: " << BeamElement() << endl;

      mutexMLSList.lock( );
      if( not ( cin && EOF!=cin.peek() ) ) { mutexMLSList.unlock(); break; }
      // Read sentence...
      uint currline = linenum++;
      cin >> lwSent >> "\n";
      cerr << "Reading sentence " << currline << ": " << lwSent << " ..." << endl;
      // Add mls to list...
      MLSs.emplace_back( );
      auto& mls = MLSs.back();
      sents.emplace_back( lwSent );
      mutexMLSList.unlock();

      if ( numtglobal == 1 ) cerr << "#" << currline;

      // For each word...
      for ( auto& w_t : lwSent ) {

        if ( numtglobal == 1 ) cerr << " " << w_t << endl;
        if ( VERBOSE ) cerr << "WORD:" << w_t << endl;

        // Create beam for current time step...
        beams[++t].clear();

        //      mutex mutexBeam;
        //      vector<thread> vtWorkers;
        //      for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&] (uint numt) {

        // For each hypothesized storestate at previous time step...
        //uint i=0; 
        for( const pair<ProbBack,BeamElement>& be_tdec1 : beams[t-1] ) { //beams[t-1] is a Beam<ProbBack,BeamElement>, so be_tdec1 is a beam item, which is a pair<ProbBack,BeamElement>. first.first is the prob in the probback, and second is the beamelement, which is a sextuple of <sign, f, e, k, j, q>
          //         if( i++%numThreads==numt ){
          double            lgpr_tdec1 = be_tdec1.first.first;      // prob of prev storestate
          const StoreState& q_tdec1    = be_tdec1.second.sixth();  // prev storestate

          //cerr << "be_tdec1.first: " << be_tdec1.first << endl;
          //if( VERBOSE>1 ) cerr << "be_tdec1.first.second: " << be_tdec1.first.second << endl;
          //if( VERBOSE>1 ) cerr << "addr be_tdec1.first.second: " << &be_tdec1.first.second << endl;
          //if( VERBOSE>1 ) cerr << "  from (" << be_tdec1.second << ")" << endl;
          
          //loop over ksAnt
          //for ( pair<ProbBack,BeamElement> biant (ProbBack(0.0,be_tdec1),BeamElement()) ; biant.first.second.something != Null ; biant = biant.first.second()) { //ej loop over antecedent beam items, with dummy BeamElement 
          //if( VERBOSE>1 ) cerr << "address of beStableDummy: " << &beStableDummy << endl;
          const BeamElement beDummy = BeamElement();
          const ProbBack pbDummy = ProbBack(0.0, be_tdec1.second);
          //const ProbBack pbDummy = ProbBack(0.0, be_tdec1.first.second);
          const pair<const BeamElement, ProbBack> biDummy( beDummy, pbDummy);
          if( VERBOSE>1 ) cerr << "bidummy second second: " << biDummy.second.second << endl;
          const pair<const BeamElement, ProbBack>* pbiAnt = &biDummy;
          //if( VERBOSE>1 ) cerr << "pbiAnt ptr to second second: " << pbiAnt->second.second << endl;
          //if( VERBOSE>1 ) cerr << "in main(): timestep t: " << t << endl;
          //
          //calculate denominator / normalizing constant over all antecedent timesteps
          double fnorm = 0.0;
          for ( int tAnt = t; tAnt>0; tAnt--, pbiAnt=&beams[tAnt].get(pbiAnt->second.second) ) { 
            //if (VERBOSE>1) cerr << "*pbiAnt: " << *pbiAnt << endl;
            if (VERBOSE>1) cerr << "pbiAnt->first: " << pbiAnt->first << endl;
            if (VERBOSE>1) cerr << "pbiAnt->second.second: " << pbiAnt->second.second << endl;
            //if (VERBOSE>1) cerr << "pbiAnt: " << pbiAnt << endl;
            const KSet ksAnt (pbiAnt->first.fourth());
            list<FPredictor> lfpredictors;  q_tdec1.calcForkPredictors( lfpredictors, ksAnt, false );  lfpredictors.emplace_back();  // add bias term //ej change
            arma::vec flogresponses = arma::zeros( matF.n_rows ); //distribution over f responses for a single antecedent features
            for ( auto& fpredr : lfpredictors ) {
              if ( fpredr.toInt() < matF.n_cols ) flogresponses += matF.col( fpredr.toInt() ); // add logprob for all indicated features. over all FEK responses.
            }
            
            arma::vec fresponses = arma::exp( flogresponses );
            double tempfnorm = arma::accu( fresponses );
            //check for underflow
            //
            if( tempfnorm == 1.0/0.0 ) {
              cerr << "WARNING: NaN for tempfnorm" << endl;
              uint ind_max=0; for(uint i=0; i<fresponses.size(); i++ ) if( fresponses(i)>fresponses(ind_max) ) ind_max=i;
              flogresponses -= flogresponses( ind_max );
              fresponses = arma::exp( flogresponses );
              tempfnorm = arma::accu( fresponses );
            }
            fnorm += tempfnorm;
          }
          pbiAnt = &biDummy; //reset pbiAnt pointer after calculating denominator
          for ( int tAnt = t; tAnt>0; tAnt--, pbiAnt=&beams[tAnt].get(pbiAnt->second.second) ) { //iterate over candidate antecedent ks, following trellis backpointers ej change for coref 
            //if( VERBOSE>1 ) cerr << "pbiAnt is biDummy: " << (pbiAnt == &biDummy) << endl;
            //if( VERBOSE>1 ) cerr << "in main(): tAnt: " << tAnt << endl;
            //if( VERBOSE>1 ) cerr << "in main(): beams[tAnt]: " << beams[tAnt] << endl;
            //cerr << "in main(): beams[tAnt].get(pbiAnt->second.second): " << beams[tAnt].get(pbiAnt->second.second) << endl;

            //if( VERBOSE>1 ) cerr << "before ksAnt init, pbiant: " << pbiAnt->second.second << endl;
            const KSet ksAnt (pbiAnt->first.fourth());

            if( VERBOSE>1 ) cerr << "ksAnt: " << ksAnt << endl;
            // Calc distrib over response for each fork predictor...
            //
            //if( VERBOSE>1 ) cerr << "after ksAnt init, pbiant: " << pbiAnt->second.second << endl;
            //if( VERBOSE>1 ) cerr << "ptr address: " << pbiAnt << endl;
            arma::vec flogresponses = arma::zeros( matF.n_rows );

            //if( VERBOSE>1 ) cerr << "after flogresponses zero init, pbiant: " << pbiAnt->second.second << endl;
            //cerr << "in main(): ksAnt: " << ksAnt << endl;
            // pbiAnt.first is a const BeamElement
            //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.fourth: " << pbiAnt->first.fourth() << endl; //" should be k
            //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.third: " << pbiAnt->first.third() << endl;   // ^@ should be extraction
            //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.second: " << pbiAnt->first.second() << endl; // 0 should be fork
            //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.first: " << pbiAnt->first.first() << endl;   // []:T^@ should be a sign
            list<FPredictor> lfpredictors;  q_tdec1.calcForkPredictors( lfpredictors, ksAnt, false );  lfpredictors.emplace_back();  // add bias term //ej change
            //if( VERBOSE>1 ) cerr << "lfpredictors  emplaced" << endl;

            //if( VERBOSE>1 ) cerr << "pbi ptr address before break: " << pbiAnt << endl;

            //if( VERBOSE>1 ) cerr << "before flogresponses accum, pbiant: " << pbiAnt->second.second << endl; //ptr dEATH
            //
            for ( auto& fpredr : lfpredictors ) {
              if ( fpredr.toInt() < matF.n_cols ) flogresponses += matF.col( fpredr.toInt() ); // add logprob for all indicated features. over all FEK responses.
              //if( VERBOSE>1 ) cerr << "lfpredictor found: " << fpredr << endl;
            }
            //if( VERBOSE>1 ) cerr << "passed fpredr loop" << endl;
            if ( VERBOSE>1 ) { for ( auto& fpredr : lfpredictors ) { cerr <<"    fpredr:"<<fpredr<<endl; } }

            //if( VERBOSE>1 ) cerr << "passed printout" << endl;
            arma::vec fresponses = arma::exp( flogresponses );
            // Calc normalization term over responses...
            //double fnorm = arma::accu( fresponses );

            //if( VERBOSE>1 ) cerr << "before overflow norm, pbiant: " << pbiAnt->second.second << endl;
            // Rescale overflowing distribs by max...
            if( fnorm == 1.0/0.0 ) {
              cerr << "WARNING: NaN for fnorm" << endl;
            }
            /*
              uint ind_max=0; for(uint i=0; i<fresponses.size(); i++ ) if( fresponses(i)>fresponses(ind_max) ) ind_max=i;
              flogresponses -= flogresponses( ind_max );
              fresponses = arma::exp( flogresponses );
              fnorm = arma::accu( fresponses );
              //            fresponses.fill( 0.0 );  fresponses( ind_max ) = 1.0;
              //            fnorm = 1.0;
            }
            */

            // For each possible lemma (context + label + prob) for preterminal of current word...
            //if( VERBOSE>1 ) cerr << "before ktprt loop, pbiant: " << pbiAnt->second.second << endl;
            //if( VERBOSE>1 ) cerr << "entering ktpr_p_t loop..." << endl;
            for ( auto& ktpr_p_t : (lexW.end()!=lexW.find(w_t)) ? lexW[w_t] : lexW[unkWord(w_t.getString().c_str())] ) {
              if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(ktpr_p_t.second) > beams[t].rbegin()->first.first ) {
                K k_p_t           = (FEATCONFIG & 8 && ktpr_p_t.first.first.getString()[2]!='y') ? K::kBot : ktpr_p_t.first.first;   // context of current preterminal
                T t_p_t           = ktpr_p_t.first.second;                               // label of current preterminal
                E e_p_t           = (t_p_t.getLastNonlocal()==N_NONE) ? 'N' : (t_p_t.getLastNonlocal()==N("-rN")) ? '0' : (t_p_t.getLastNonlocal().isArg()) ? t_p_t.getArity()+'1' : 'M';
                double probwgivkl = ktpr_p_t.second;                                     // probability of current word given current preterminal

                if ( VERBOSE>1 ) cerr << "     W " << k_p_t << " " << t_p_t << " : " << w_t << " = " << probwgivkl << endl; 

                // For each possible no-fork or fork decision...
                for ( auto& f : {0,1} ) {
                  double scoreFork = ( FResponse::exists(f,e_p_t,k_p_t) ) ? fresponses(FResponse(f,e_p_t,k_p_t).toInt()) : 1.0 ;
                  if ( VERBOSE>1 ) cerr << "      F ... : " << f << " " << e_p_t << " " << k_p_t << " = " << (scoreFork / fnorm) << endl;
                  //if( VERBOSE>1 ) cerr << "fork pbiant: " << pbiAnt->second.second << endl;

                  // If preterminal prob is nonzero...
                  PPredictor ppredictor = q_tdec1.calcPretrmTypeCondition(f,e_p_t,k_p_t);
                  if ( VERBOSE>1 ) cerr << "      P " << ppredictor << "..." << endl;
                  if ( modP.end()!=modP.find(ppredictor) && modP[ppredictor].end()!=modP[ppredictor].find(t_p_t) ) {
                    if ( VERBOSE>1 ) cerr << "      P " << ppredictor << " : " << t_p_t << " = " << modP[ppredictor][t_p_t] << endl;
                    // Calc probability for fork phase...
                    double probFork = (scoreFork / fnorm) * modP[ppredictor][t_p_t] * probwgivkl;
                    if ( VERBOSE>1 ) cerr << "      f: f" << f << "&" << e_p_t << "&" << k_p_t << " " << scoreFork << " / " << fnorm << " * " << modP[ppredictor][t_p_t] << " * " << probwgivkl << " = " << probFork << endl;
                    Sign aPretrm;  aPretrm.first().emplace_back(k_p_t);  aPretrm.second() = t_p_t;  aPretrm.third() = S_A;          // aPretrm (pos tag)
                    for (auto& ant : ksAnt) {
                      //don't add ant if Bot
                      aPretrm.first().emplace_back(ant); // coref change to add antecedent Ks to aPretrm 
                    }
                    const LeftChildSign aLchild( q_tdec1, f, e_p_t, aPretrm );
                    list<JPredictor> ljpredictors; q_tdec1.calcJoinPredictors( ljpredictors, f, e_p_t, aLchild, false ); // predictors for join
                    ljpredictors.emplace_back();                                                                  // add bias
                    arma::vec jlogresponses = arma::zeros( matJ.n_rows );
                    for ( auto& jpredr : ljpredictors ) if ( jpredr.toInt() < matJ.n_cols ) jlogresponses += matJ.col( jpredr.toInt() );
                    arma::vec jresponses = arma::exp( jlogresponses );
                    double jnorm = arma::accu( jresponses );  // 0.0;                                           // join normalization term (denominator)

                    // Replace overflowing distribs by max...
                    if( jnorm == 1.0/0.0 ) {
                      uint ind_max=0; for(uint i=0; i<jlogresponses.size(); i++ ) if ( jlogresponses(i)>jlogresponses(ind_max) ) ind_max=i;
                      jlogresponses -= jlogresponses( ind_max );
                      jresponses = arma::exp( jlogresponses );
                      jnorm = arma::accu( jresponses );
                      //                    jresponses.fill( 0.0 );  jresponses( ind_max ) = 1.0;
                      //                    jnorm = 1.0;
                    }

                    // For each possible no-join or join decision, and operator decisions...
                    for( JResponse jresponse; jresponse<JResponse::getDomain().getSize(); ++jresponse ) {
                      if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(jresponses[jresponse.toInt()]/jnorm) > beams[t].rbegin()->first.first ) {
                        J j   = jresponse.getJoin();
                        E e   = jresponse.getE();
                        O opL = jresponse.getLOp();
                        O opR = jresponse.getROp();
                        double probJoin = jresponses[jresponse.toInt()] / jnorm;
                        if ( VERBOSE>1 ) cerr << "       J ... " << " : " << jresponse << " = " << probJoin << endl;

                        // For each possible apex category label...
                        APredictor apredictor = q_tdec1.calcApexTypeCondition( f, j, e_p_t, e, opL, aLchild );  // save apredictor for use in prob calc
                        if ( VERBOSE>1 ) cerr << "         A " << apredictor << "..." << endl;
                        if ( modA.end()!=modA.find(apredictor) ) {
                          for ( auto& tpA : modA[apredictor] ) {
                            if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) > beams[t].rbegin()->first.first ) {
                              if ( VERBOSE>1 ) cerr << "         A " << apredictor << " : " << tpA.first << " = " << tpA.second << endl;

                              // For each possible brink category label...
                              BPredictor bpredictor = q_tdec1.calcBrinkTypeCondition( f, j, e_p_t, e, opL, opR, tpA.first, aLchild );  // bpredictor for prob calc
                              if ( VERBOSE>1 ) cerr << "          B " << bpredictor << "..." << endl;
                              if ( modB.end()!=modB.find(bpredictor) ) {
                                for ( auto& tpB : modB[bpredictor] ) {
                                  if ( VERBOSE>1 ) cerr << "          B " << bpredictor << " : " << tpB.first << " = " << tpB.second << endl;
                                  //                            lock_guard<mutex> guard( mutexBeam );
                                  if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second) > beams[t].rbegin()->first.first ) {

                                    // Calculate probability and storestate and add to beam...
                                    StoreState ss( q_tdec1, f, j, e_p_t, e, opL, opR, tpA.first, tpB.first, aPretrm, aLchild );
                                    if( (t<lwSent.size() && ss.size()>0) || (t==lwSent.size() && ss.size()==0) ) {
                                      beams[t].tryAdd( BeamElement( aPretrm, f,e_p_t,k_p_t, jresponse, ss ), ProbBack( lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second), be_tdec1.second ) );
                                      if( VERBOSE>1 ) cerr << "                send (" << be_tdec1.second << ") to (" << ss << ") with "
                                        << (lgpr_tdec1 + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second)) << endl;
                                    } //if t<lwSent
                                  } //if beams[t].size()<BEAM_WIDTH 
                                } //for tpB
                              } //if modB.end()!=modB.find(bpredictor)
                            } //if beams[t].size()<BEAM_WIDTH
                          } // for tpA : modA
                        } //if modA.end()!=modA.find(apredictor)
                      } // if beams
                    } // for jresponse
                  } // if modP
                  //if( VERBOSE>1 ) cerr << "finished modP. iterating f..." << endl;
                  //if( VERBOSE>1 ) cerr << "after modP, pbiant->second.second: " << pbiAnt->second.second << endl;
                } // for f : {0,1}
              //if( VERBOSE>1 ) cerr << "finished iterating f.  iterating kptr_p_t..." << endl;
              //if( VERBOSE>1 ) cerr << "after f, pbiant->second.second: " << pbiAnt->second.second << endl;
              } // if beamsize
            } // for ktpr_p_t
            //if( VERBOSE>1 ) cerr << "finished iterating kptr_p_t. iterating tant previous timesteps..." << endl;
            //if( VERBOSE>1 ) cerr << "t-1: " << t-1 << " sizeof beams[t-1]: " << beams[t-1].size() << endl;
            //if( VERBOSE>1 ) cerr << "tAnt: " << tAnt << endl;
            //if( VERBOSE>1 ) cerr << "after kptr, pbiant->second.second: " << pbiAnt->second.second << endl;
            //cerr << "beams[0].get(pbiant->second.second): " << beams[0].get(pbiAnt->second.second) << endl;
          } //for tant 
          //if( VERBOSE>1 ) cerr << "finished antecedent timesteps, iterating beam item..." << endl;
        } //for be_tdec1 : beams[t-1] 
        //if( VERBOSE>1 ) cerr << "finished beam items, iterating word..." << endl; 
          //      }, numtglobal ));

          //      for( auto& w : vtWorkers ) w.join();

        // Write output...
        if ( numtglobal == 1 ) { cerr << "beamsize at t: (" << beams[t].size() << ")"; }
        if ( VERBOSE ) { cerr << "trying to print beam..." << endl; cerr << beams[t] << endl; }
      } // w_t : lwSent
      if( VERBOSE>1 ) cerr << "finished words, iterating sentence..." << endl;
      if ( numtglobal == 1 ) cerr << endl;
      if ( VERBOSE ) cerr << "MLS" << endl;

      //DelimitedList<psX,pair<BeamElement,ProbBack>,psLine,psX> mls;
      { lock_guard<mutex> guard( mutexMLSList );
        if( numtglobal > 1 ) cerr << "Finished line " << currline << " (" << beams[t].size() << ")..." << endl;
        beams.setMostLikelySequence( mls );
      }
      /*
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
      */

      { lock_guard<mutex> guard( mutexMLSList );
  //        for( auto& mls : MLSs )// cerr<<"on list: "<<mls.size()<<endl;
  //          for( auto& be : mls ) cerr<<"on list: " << be.first << endl;
        while( MLSs.size()>0 && MLSs.front().size()>0 ) {
          auto& mls = MLSs.front( );
          int u=1; auto ibe=next(mls.begin()); auto iw=sents.front().begin(); for( ; ibe!=mls.end() && iw!=sents.front().end(); ibe++, iw++, u++ ) {
            cout << *iw << " " << ibe->first;
            cout << endl;
          } //for ibe
          MLSs.pop_front();
          sents.pop_front();
        } //while MLSs.size()>0
      } //lock_guard mutex mutexMLSList
    } // while True
  }, numtglobal )); //end lambda function def for pushback 

  for( auto& w : vtWorkers ) w.join();

} //main 

