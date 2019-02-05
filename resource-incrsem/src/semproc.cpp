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
#include <chrono>
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

uint BEAM_WIDTH      = 1000;
uint VERBOSE         = 0;
uint OUTPUT_MEASURES = 0;
// bool USE_COREF = true;
////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<T> P;
typedef Delimited<T> A;
typedef Delimited<T> B;

////////////////////////////////////////////////////////////////////////////////

class Trellis : public vector<Beam<HiddState>> {
  // private:
  //  DelimitedList<psX,pair<HiddState,ProbBack>,psLine,psX> lbe;
  public:
    Trellis ( ) : vector<Beam<HiddState>>() { reserve(100); }
    Beam<HiddState>& operator[] ( uint i ) { if ( i==size() ) emplace_back(BEAM_WIDTH); return vector<Beam<HiddState>>::operator[](i); }
    void setMostLikelySequence ( DelimitedList<psX,BeamElement<HiddState>,psLine,psX>& lbe ) {
      static StoreState ssLongFail( StoreState(), 1, 0, EVar::eNil, EVar::eNil, 'N', 'I', "FAIL", "FAIL", Sign(ksBot,"FAIL",0), Sign(ksBot,"FAIL",0) ); //fork, nojoin
      lbe.clear(); if( back().size()>0 ) lbe.push_front( *back().begin() );
      if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( lbe.front().getBack() );
      if( lbe.size()>0 ) lbe.emplace_back( BeamElement<HiddState>() );
      cerr << "lbe.size(): " << lbe.size() << endl;
      // If parse fails...
      if( lbe.size()==0 ) {
        cerr << "parse failed (lbe.size() = 0) " << "size(): " << size() << endl;
        // Print a right branching structure...
        for( int t=size()-2; t>=0; t-- ) { 
          lbe.push_front( BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(ksBot,"FAIL",0), 1, EVar::eNil, K::kBot, JResponse(1,EVar::eNil,'N','I'), ssLongFail ) ) ); // fork and join
        }
        cerr << "size of lbe after push_fronts: " << lbe.size() << endl;
        lbe.front() = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(ksBot,"FAIL",0), 1, EVar::eNil, K::kBot, JResponse(0,EVar::eNil,'N','I'), ssLongFail ) );                    // front: fork no-join
        lbe.back( ) = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(ksBot,"FAIL",0), 0, EVar::eNil, K::kBot, JResponse(1,EVar::eNil,'N','I'), StoreState() ) );                  // back: join no-fork
        cerr << "size of lbe after front and back assignments: " << lbe.size() << endl;
        if( size()==2 ) {  //special case if single word, fork and join
          cerr << "assigning front of fail lbe" << endl;
          lbe.front() = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(ksBot,"FAIL",0), 1, EVar::eNil, K::kBot, JResponse(1,EVar::eNil,'N','I'), StoreState() ) );  // unary case: fork and join
        }
        // Add dummy element (not sure why this is needed)...
        lbe.push_front( BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(ksBot,"FAIL",0), 0, EVar::eNil, K::kBot, JResponse(0,EVar::eNil,'N','I'), StoreState() ) ) ); // no-fork, no-join?
        //start experiment - next two lines switch front element to nofork,join, add additional dummy at rear
        //TODO to revert, comment out next two, comment in pushfront above
        //lbe.push_front( BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(ksBot,"FAIL",0), 1, EVar::eNil, K::kBot, JResponse(1,EVar::eNil,'N','I'), ssLongFail ) ) );
        lbe.emplace_back( BeamElement<HiddState>() );
        //end epxeriment

        cerr << "size of lbe after dummy push_front: " << lbe.size() << endl;
        cerr<<"parse failed"<<endl;
        // does lbe here consist of a single sentence or of the whole article?
      }
      // For each element of MLE after first dummy element...
      for ( auto& be : lbe ) { cerr << "beam element hidd: " << be.getHidd() << endl; } //TODO confirm includes all words, count initial/final dummies
      int u=-1; for( auto& be : lbe ) if( ++u>0 and u<int(size()) ) {
        // Calc surprisal as diff in exp of beam totals of successive elements, minus constant...
        double probPrevTot = 0.0;
        double probCurrTot = 0.0;
        for( auto& beP : at(u-1) ) probPrevTot += exp( beP.getProb() - at(u-1).begin()->getProb() );
        for( auto& beC : at(u  ) ) probCurrTot += exp( beC.getProb() - at(u-1).begin()->getProb() ); 
        be.setProb() = log2(probPrevTot) - log2(probCurrTot);     // store surp into prob field of beam item
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

vector<const char*> vpsInts = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  uint numThreads = 1;

  // Define model structures...
  arma::mat matF;
  arma::mat matJ;
  arma::mat matN;
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
    list<DelimitedTrip<psX,NPredictor,psSpcColonSpc,Delimited<NResponse>,psSpcEqualsSpc,Delimited<double>,psX>> lN;

    lA.emplace_back( DelimitedTrip<psX,APredictor,psSpcColonSpc,A,psSpcEqualsSpc,Delimited<double>,psX>(APredictor(1,0,1,EVar::eNil,'S',T("T"),T("-")),A("-"),1.0) );      // should be T("S")
    lB.emplace_back( DelimitedTrip<psX,BPredictor,psSpcColonSpc,B,psSpcEqualsSpc,Delimited<double>,psX>(BPredictor(1,0,1,EVar::eNil,'S','1',T("-"),T("S")),B("T"),1.0) );

    // For each command-line flag or model file...
    for ( int a=1; a<nArgs; a++ ) {
      if      ( 0==strcmp(argv[a],"-v") ) VERBOSE = 1;
      else if ( 0==strcmp(argv[a],"-V") ) VERBOSE = 2;
      else if ( 0==strncmp(argv[a],"-p",2) ) numThreads = atoi(argv[a]+2);
      else if ( 0==strncmp(argv[a],"-b",2) ) BEAM_WIDTH = atoi(argv[a]+2);
      else if ( 0==strcmp(argv[a],"-c") ) OUTPUT_MEASURES = 1;
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
          if ( fin.peek()=='N' ) fin >> "N " >> *lN.emplace(lN.end()) >> "\n";
          if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
        }
        cerr << "Model " << argv[a] << " loaded." << endl;
      }
    } //closes for int a=1

    // Populate model structures...
    matF = arma::zeros( FResponse::getDomain().getSize(), FPredictor::getDomainSize() );
    matJ = arma::zeros( JResponse::getDomain().getSize(), JPredictor::getDomainSize() );
    matN = arma::zeros( NResponse::getDomain().getSize(), NPredictor::getDomainSize() );
    for ( auto& prw : lF ) matF( prw.second().toInt(), prw.first().toInt() ) = prw.third();
    for ( auto& prw : lP ) modP[prw.first()][prw.second()] = prw.third();
    for ( auto& prw : lW ) lexW[prw.second()].emplace_back(prw.first(),prw.third());
    for ( auto& prw : lJ ) matJ( prw.second().toInt(), prw.first().toInt() ) = prw.third();
    for ( auto& prn : lN ) matN( prn.second().toInt(), prn.first().toInt() ) = prn.third(); //i,jth cell of matrix gets populated with value
    for ( auto& prw : lA ) modA[prw.first()][prw.second()] = prw.third();
    for ( auto& prw : lB ) modB[prw.first()][prw.second()] = prw.third();
  } //closes define model lists

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
  } //closes for auto& entry : lexW

  cerr<<"Models ready."<<endl;

  //list<DelimitedList<psX,BeamElement<HiddState>,psLine,psX>> MLSs;
  //list<DelimitedList<psX,ObsWord,psSpace,psX>> sents;
  mutex mutexMLSList;
  vector<thread> vtWorkers;  vtWorkers.reserve( numThreads );
  uint linenum = 0;

  if( OUTPUT_MEASURES ) cout << "word pos f j store totsurp" << endl;
  
  // For each line in stdin...
  //  for ( int linenum=1; cin && EOF!=cin.peek(); linenum++ ) {
  //for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&MLSs,&sents,&mutexMLSList,&linenum,numThreads,matN,matF,modP,lexW,matJ,modA,modB] (uint numt) {
  list<list<DelimitedList<psX,ObsWord,psSpace,psX>>> articles; //list of list of sents. each list of sents is an article.
  list<list<DelimitedList<psX,BeamElement<HiddState>,psLine,psX>>> articleMLSs; //list of MLSs

  // loop over threads (each thread gets an article)
  for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&articleMLSs,&articles,&mutexMLSList,&linenum,numThreads,matN,matF,modP,lexW,matJ,modA,modB] (uint numt) {
    auto tpLastReport = chrono::high_resolution_clock::now();

    while( true ) {
      // Read in your worker thread's article in this lock block
      mutexMLSList.lock( );
      if( not ( cin && EOF!=cin.peek() ) ) { mutexMLSList.unlock(); break; }

      uint currline = linenum; 
      cerr << "Worker: " << numt << " attempting to emplace back of articles..." << endl;
      articles.emplace_back(); 
      cerr << "Worker: " << numt << " attempting to assign sents as last article..." << endl;
      auto& sents = articles.back(); //a specific article becomes the thread's sents //returns reference
      cerr << "Worker: " << numt << " attempting to emplace back of articleMLSs..." << endl;
      articleMLSs.emplace_back();
      cerr << "Worker: " << numt << " attempting to assign MLSs as last articleMLS..." << endl;
      auto& MLSs = articleMLSs.back();
      cerr << "Worker: " << numt << " finished initial assignments" << endl;

      DelimitedList<psX,ObsWord,psSpace,psX> articleDelim; // !article should be consumed between sentence reads
      //loop over sentences in an article
      cin >> articleDelim >> "\n"; //consume !ARTICLE

      while (cin.peek()!='!' and cin.peek()!=EOF) {
        // Read sentence...
        linenum++; //updates linenum for when handing off to other thread
        //uint currline = linenum++; //calculate this as looping through articlesents
        DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // init input list for each iteration - otherwise cin just keeps appending to existing lwSent
        cin >> lwSent >> "\n";
        cerr << "Worker: " << numt << " Reading sentence " << linenum << ": " << lwSent << " EOS" << endl;
        sents.emplace_back( lwSent );
        cerr << "Worker: " << numt << " successfully read sentence " << linenum << endl;
      }
      mutexMLSList.unlock();

      if ( numThreads == 1 ) cerr << "#" << currline;

      DelimitedList<psX,BeamElement<HiddState>,psLine,psX> lbeWholeArticle;
      lbeWholeArticle.emplace_back(); //create null beamElement at start of article

      for (auto& lwSent : sents) {
        currline++;

        // Add mls to list...
        MLSs.emplace_back( ); //establish placeholder for mls for this specific sentence
        auto& mls = MLSs.back();

        Trellis   beams;  // sequence of beams - each beam is hypotheses at a given timestep
        uint      t=0;    // time step

        // Allocate space in beams to avoid reallocation...
        // Create initial beam element...
        //beams[0].tryAdd( HiddState(), ProbBack<HiddState>() );
        //TODO see if resetting each sentences to use zero prob instead of last prob avoids underflow
        lbeWholeArticle.back().setProb() = 0.0;
        beams[0].tryAdd( lbeWholeArticle.back().getHidd(), lbeWholeArticle.back().getProbBack() );
        //beams[0].tryAdd( lbeWholeArticle.back().getHidd(), 0 ); //nope, cant do int. maybe needs probback

        //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " testing access to articles: " << articles.size() << " articles present" << endl;}
        //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " testing access to sents: " << sents.size() << " sents found" << endl;}
        //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " testing access to lwSent: " << lwSent << endl;}
        //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " starting sentence loop..." << endl;}

        // For each word... 
        for ( auto& w_t : lwSent ) {
          //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " beginning word loop with word: " << w_t << endl;}
          //DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // input list
          if ( numThreads == 1 ) cerr << " " << w_t;
          if ( VERBOSE ) cout << "WORD:" << w_t << endl;

          // Create beam for current time step...
          beams[++t].clear();

          //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " just cleared beam..." << w_t << endl;}
          // For each hypothesized storestate at previous time step...
          //uint i=0; for( auto& be_tdec1 : beams[t-1] ) {
          
          for( const BeamElement<HiddState>& be_tdec1 : beams[t-1] ) { //beams[t-1] is a Beam<ProbBack,BeamElement>, so be_tdec1 is a beam item, which is a pair<ProbBack,BeamElement>. first.first is the prob in the probback, and second is the beamelement, which is a sextuple of <sign, f, e, k, j, q>
            //         if( i++%numThreads==numt ){

            double            lgpr_tdec1 = be_tdec1.getProb(); // logprob of prev storestate
            const StoreState& q_tdec1    = be_tdec1.getHidd().sixth();  // prev storestate

            if( VERBOSE>1 ) cout << "  from (" << be_tdec1.getHidd() << ")" << endl;

            const ProbBack<HiddState> pbDummy = ProbBack<HiddState>(0.0, be_tdec1); //dummy element for most recent timestep
            const HiddState hsDummy = HiddState(Sign(ksTop,T(),S()),F(),EVar(),K(),JResponse(),StoreState(),0 ); //dummy hidden state with kTop semantics 
            const BeamElement<HiddState> beDummy = BeamElement<HiddState>(pbDummy, hsDummy); //at timestep t, represents null antecedent 
            //const ProbBack pbDummy = ProbBack(0.0, be_tdec1.first.second);
            //const BeamElement<HiddState> biDummy( beDummy, pbDummy);
            //if( VERBOSE>1 ) cerr << "bidummy second second: " << biDummy.second.second << endl;
            const BeamElement<HiddState>* pbeAnt = &beDummy;
            //if( VERBOSE>1 ) cerr << "pbiAnt ptr to second second: " << pbiAnt->second.second << endl;
            //if( VERBOSE>1 ) cerr << "in main(): timestep t: " << t << endl;
            //
            //calculate denominator / normalizing constant over all antecedent timesteps
            double fnorm = 0.0; 
            //for ( int tAnt = t; tAnt>0; tAnt--, pbiAnt=&beams[tAnt].get(pbiAnt->second.second) ) { 
            double ndenom = 0.0;
            //for ( int tAnt = t; tAnt>((USE_COREF) ? 0 : t-1); tAnt--, pbeAnt = &pbeAnt->getBack()) { //denominator

            //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " starting denom loop..." << w_t << endl;}
            //for ( int tAnt = t; tAnt>0; tAnt--, pbeAnt = &pbeAnt->getBack()) { //denominator
            for ( int tAnt = t; &pbeAnt->getBack() != &BeamElement<HiddState>::beStableDummy; tAnt--, pbeAnt = &pbeAnt->getBack()) { //denominator
              //if (VERBOSE>1) cerr << "*pbiAnt: " << *pbiAnt << endl;
              //if (VERBOSE>1) cerr << "pbiAnt->first: " << pbiAnt->first << endl;
              //if (VERBOSE>1) cerr << "pbiAnt->second.second: " << pbiAnt->second.second << endl;
              //if (VERBOSE>1) cerr << "pbiAnt: " << pbiAnt << endl;
              //const KSet ksAnt (pbiAnt->first.fourth());
              const KSet ksAnt (pbeAnt->getHidd().getPrtrm().getKSet()); 
              //if (t < 2) { 
                //cerr << "t: " << t << ", tAnt: " << tAnt << ", neq test: " << (&pbeAnt->getBack() != &BeamElement<HiddState>::beStableDummy) << endl;
                //cerr << "pbeAnt ptr: " << pbeAnt << endl;
                //cerr << "pbeAnt->getHidd(): " << pbeAnt->getHidd() << endl;
                //cerr << "considering ksAnt: " << ksAnt << endl;
              //}
              //{ lock_guard<mutex> guard( mutexMLSList ); cerr << "Worker: " << numt << " got ksAnt" << ksAnt << endl;}
              bool corefON = (tAnt==t) ? 0 : 1;
              list<NPredictor> lnpredictors; q_tdec1.calcNPredictors( lnpredictors, pbeAnt->getHidd().getPrtrm(), corefON); 
              arma::vec nlogresponses = arma::zeros( matN.n_rows ); //rows are n outcomes 1 or 0 (coreferent or not)
              for ( auto& npredr : lnpredictors ) {
                if ( npredr.toInt() < matN.n_cols ) {
                  nlogresponses += matN.col( npredr.toInt() );  //cols are predictors, where each col has values for 1 or 0
                }
              }

              //arma::vec nresponses = arma::exp( nlogresponses );
              //double nnorm = arma::accu( nresponses );  // 0.0;                                           // join normalization term (denominator)
              ndenom += exp(nlogresponses(NResponse("1").toInt())-nlogresponses(NResponse("0").toInt()));
              //list<FPredictor> lfpredictors;  q_tdec1.calcForkPredictors( lfpredictors, ksAnt, false );  lfpredictors.emplace_back();  // add bias term //ej change
              // Calc distrib over response for each fork predictor...
              //arma::vec flogresponses = arma::zeros( matF.n_rows );            //distribution over f responses for a single antecedent features
              //for ( auto& fpredr : lfpredictors ) if ( fpredr.toInt() < matF.n_cols ) flogresponses += matF.col( fpredr.toInt() ); // add logprob for all indicated features. over all FEK responses.
              //if ( VERBOSE>1 ) for ( auto& fpredr : lfpredictors ) cout<<"    fpredr:"<<fpredr<<endl;
              //arma::vec fresponses = arma::exp( flogresponses );
              //double tempfnorm = arma::accu( fresponses );
              // Calc normalization term over responses...
              //double fnorm = arma::accu( fresponses );

              /*
              // Rescale overflowing distribs by max...
              if( tempfnorm == 1.0/0.0 ) {
                cerr << "WARNING: NaN for tempfnorm" << endl;
                uint ind_max=0; for( uint i=0; i<fresponses.size(); i++ ) if( fresponses(i)>fresponses(ind_max) ) ind_max=i;
                flogresponses -= flogresponses( ind_max );
                fresponses = arma::exp( flogresponses );
                tempfnorm = arma::accu( fresponses );
                //            fresponses.fill( 0.0 );  fresponses( ind_max ) = 1.0;
                //            fnorm = 1.0;
              } //closes if tempfnorm
              fnorm += tempfnorm;
              */
            } //closes for tAnt
            pbeAnt = &beDummy; //reset pbiAnt pointer after calculating denominator

            //for ( int tAnt = t; tAnt>0; tAnt--, pbiAnt=&beams[tAnt].get(pbiAnt->second.second) ) { //iterate over candidate antecedent ks, following trellis backpointers ej change for coref 
            //for ( int tAnt = t; tAnt>((USE_COREF) ? 0 : t-1); tAnt--, pbeAnt = &pbeAnt->getBack()) { //iterate over candidate antecedent ks, following trellis backpointers ej change for coref 
            //
            //{ lock_guard<mutex> guard( mutexMLSList );   cerr << "Worker: " << numt << " starting numerator loop..." << w_t << endl;}
            for ( int tAnt = t; &pbeAnt->getBack() != &BeamElement<HiddState>::beStableDummy; tAnt--, pbeAnt = &pbeAnt->getBack()) { //numerator, iterate over candidate antecedent ks, following trellis backpointers. 
              //if( VERBOSE>1 ) cerr << "pbiAnt is biDummy: " << (pbiAnt == &biDummy) << endl;
              //if( VERBOSE>1 ) cerr << "in main(): tAnt: " << tAnt << endl;
              //if( VERBOSE>1 ) cerr << "in main(): beams[tAnt]: " << beams[tAnt] << endl;
              //cerr << "in main(): beams[tAnt].get(pbiAnt->second.second): " << beams[tAnt].get(pbiAnt->second.second) << endl;

              //if( VERBOSE>1 ) cerr << "before ksAnt init, pbiant: " << pbiAnt->second.second << endl;
              const KSet ksAnt (pbeAnt->getHidd().getPrtrm().getKSet());
              //if( VERBOSE>1 ) cerr << "ksAnt: " << ksAnt << endl;

              //Calculate antecedent N model predictors 
              bool corefON = (tAnt==t) ? 0 : 1;
              list<NPredictor> lnpredictors; q_tdec1.calcNPredictors( lnpredictors, pbeAnt->getHidd().getPrtrm(), corefON); //calcNPredictors takes list of npreds (reference) and candidate Sign (reference)
              //lnpredictors.emplace_back();                                             // add bias. don't need because included in StoreState calcNPredictors

              if ( VERBOSE>1 ) { for ( auto& npredr : lnpredictors ) { cout <<"   npredr:"<<npredr<<endl; } }
              arma::vec nlogresponses = arma::zeros( matN.n_rows );
              for ( auto& npredr : lnpredictors ) {
                  //if (VERBOSE>1) { cout << npredr << " " << npredr.toInt() << endl; }
                if ( npredr.toInt() < matN.n_cols ) { 
                  nlogresponses += matN.col( npredr.toInt() ); 
                  //if (VERBOSE>1) { cout << npredr << " " << npredr.toInt() << " matN.n_cols:" << matN.n_cols << " logprob: " << matN.col( npredr.toInt())(NResponse("1").toInt()) << endl; }
                }
              }
              double numerator = exp(nlogresponses(NResponse("1").toInt()) - nlogresponses(NResponse("0").toInt()));
              double nprob = numerator / ndenom;

              if ( VERBOSE>1 ) cout << "   N ... : 1 = " << numerator << "/" << ndenom << "=" << nprob << endl;
              //arma::vec nresponses = arma::exp( nlogresponses );
              //double nnorm = arma::accu( nresponses );  // 0.0;                                           // join normalization term (denominator)

              // Calc distrib over response for each fork predictor...
              //
              //if( VERBOSE>1 ) cerr << "after ksAnt init, pbiant: " << pbiAnt->second.second << endl;
              //if( VERBOSE>1 ) cerr << "ptr address: " << pbiAnt << endl;
              //arma::vec flogresponses = arma::zeros( matF.n_rows );

              //if( VERBOSE>1 ) cerr << "after flogresponses zero init, pbiant: " << pbiAnt->second.second << endl;
              //cerr << "in main(): ksAnt: " << ksAnt << endl;
              // pbiAnt.first is a const BeamElement
              //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.fourth: " << pbiAnt->first.fourth() << endl; //" should be k
              //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.third: " << pbiAnt->first.third() << endl;   // ^@ should be extraction
              //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.second: " << pbiAnt->first.second() << endl; // 0 should be fork
              //if( VERBOSE>1 ) cerr << "in main(): pbiAnt.first.first: " << pbiAnt->first.first() << endl;   // []:T^@ should be a sign
              //list<FPredictor> lfpredictors;  q_tdec1.calcForkPredictors( lfpredictors, ksAnt, false );  lfpredictors.emplace_back();  // add bias term //ej change
              //if( VERBOSE>1 ) cerr << "lfpredictors  emplaced" << endl;

              //if( VERBOSE>1 ) cerr << "pbi ptr address before break: " << pbiAnt << endl;

              //if( VERBOSE>1 ) cerr << "before flogresponses accum, pbiant: " << pbiAnt->second.second << endl; //ptr dEATH
              //
              list<FPredictor> lfpredictors;  
              q_tdec1.calcForkPredictors( lfpredictors, ksAnt, !corefON, false ); 
              lfpredictors.emplace_back();  // add bias term
              // Calc distrib over response for each fork predictor...
              arma::vec flogresponses = arma::zeros( matF.n_rows );            //distribution over f responses for a single antecedent features
              for ( auto& fpredr : lfpredictors ) if ( fpredr.toInt() < matF.n_cols ) flogresponses += matF.col( fpredr.toInt() ); // add logprob for all indicated features. over all FEK responses.
              //if ( VERBOSE>1 ) for ( auto& fpredr : lfpredictors ) cout<<"    fpredr:"<<fpredr<<endl;
              arma::vec fresponses = arma::exp( flogresponses );
              fnorm = arma::accu( fresponses );
              //if( VERBOSE>1 ) cerr << "lfpredictor found: " << fpredr << endl;
              //if( VERBOSE>1 ) cerr << "passed fpredr loop" << endl;
              if ( VERBOSE>1 ) { for ( auto& fpredr : lfpredictors ) { cout <<"    fpredr:"<<fpredr<<endl; } }

              //if( VERBOSE>1 ) cerr << "passed printout" << endl;
              //arma::vec fresponses = arma::exp( flogresponses );
              // Calc normalization term over responses...
              //double fnorm = arma::accu( fresponses );
  //if( VERBOSE>1 ) cerr << "before overflow norm, pbiant: " << pbiAnt->second.second << endl; // Rescale overflowing distribs by max...  if( fnorm == 1.0/0.0 ) { cerr << "WARNING: NaN for fnorm" << endl; } //} close for loop over antecedents here?
              // For each possible lemma (context + label + prob) for preterminal of current word...
              if( lexW.end() == lexW.find(unkWord(w_t.getString().c_str())) ) cerr<<"ERROR: unable to find unk form: "<<unkWord(w_t.getString().c_str())<<endl;
              for ( auto& ektpr_p_t : (lexW.end()!=lexW.find(w_t)) ? lexW.find(w_t)->second : lexW.find(unkWord(w_t.getString().c_str()))->second ) {
                if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(ektpr_p_t.second) > beams[t].rbegin()->getProb() ) {
                  EVar  e_p_t       = ektpr_p_t.first.first();
                  K     k_p_t       = (FEATCONFIG & 8 && ektpr_p_t.first.second().getString()[2]!='y') ? K::kBot : ektpr_p_t.first.second();   // context of current preterminal
                  T     t_p_t       = ektpr_p_t.first.third();                               // label of current preterminal
                  //              EVar  e_p_t       = (t_p_t.getLastNonlocal()==N_NONE) ? EVar::eNil : (t_p_t.getLastNonlocal()==N("-rN")) ? "0" : (t_p_t.getLastNonlocal().isArg()) ? vpsInts[t_p_t.getArity()+1] : "M";
                  double probwgivkl = ektpr_p_t.second;                                     // probability of current word given current preterminal

                  if ( VERBOSE>1 ) cout << "     W " << e_p_t << " " << k_p_t << " " << t_p_t << " : " << w_t << " = " << probwgivkl << endl;

                  // For each possible no-fork or fork decision...
                  for ( auto& f : {0,1} ) {
                    if( FResponse::exists(f,e_p_t,k_p_t) && FResponse(f,e_p_t,k_p_t).toInt() >= int(fresponses.size()) ) cerr<<"ERROR: unable to find fresponse "<<FResponse(f,e_p_t,k_p_t)<<endl;
                    double scoreFork = ( FResponse::exists(f,e_p_t,k_p_t) ) ? fresponses(FResponse(f,e_p_t,k_p_t).toInt()) : 1.0 ;
                    if ( VERBOSE>1 ) cout << "      F ... : " << f << " " << e_p_t << " " << k_p_t << " = " << (scoreFork / fnorm) << endl;

                    if( chrono::high_resolution_clock::now() > tpLastReport + chrono::minutes(1) ) {
                      tpLastReport = chrono::high_resolution_clock::now();
                      lock_guard<mutex> guard( mutexMLSList );
                      cerr << "WORKER " << numt << ": SENT " << currline << " WORD " << t << " FROM " << be_tdec1.getHidd() << " PRED " << ektpr_p_t << endl;
                    } //closes if chrono

                    // If preterminal prob is nonzero...
                    PPredictor ppredictor = q_tdec1.calcPretrmTypeCondition(f,e_p_t,k_p_t);
                    if ( VERBOSE>1 ) cout << "      P " << ppredictor << " : " << t_p_t << "...?" << endl;
                    if ( modP.end()!=modP.find(ppredictor) && modP.find(ppredictor)->second.end()!=modP.find(ppredictor)->second.find(t_p_t) ) {

                      if ( VERBOSE>1 ) cout << "      P " << ppredictor << " : " << t_p_t << " = " << modP.find(ppredictor)->second.find(t_p_t)->second << endl;

                      // Calc probability for fork phase...
                      double probFork = (scoreFork / fnorm) * modP.find(ppredictor)->second.find(t_p_t)->second * probwgivkl;
                      if ( VERBOSE>1 ) cout << "      f: f" << f << "&" << e_p_t << "&" << k_p_t << " " << scoreFork << " / " << fnorm << " * " << modP.find(ppredictor)->second.find(t_p_t)->second << " * " << probwgivkl << " = " << probFork << endl;

                      Sign aPretrm;  aPretrm.first().emplace_back(k_p_t);  
                      for (auto& k : ksAnt) if (k != K::kTop) aPretrm.first().emplace_back(k); // add antecedent contexts
                      aPretrm.second() = t_p_t;  aPretrm.third() = S_A;          // aPretrm (pos tag)
                      const LeftChildSign aLchild( q_tdec1, f, e_p_t, aPretrm );
                      list<JPredictor> ljpredictors; q_tdec1.calcJoinPredictors( ljpredictors, f, e_p_t, aLchild, false ); // predictors for join
                      ljpredictors.emplace_back();                                                                  // add bias
                      arma::vec jlogresponses = arma::zeros( matJ.n_rows );
                      for ( auto& jpredr : ljpredictors ) if ( jpredr.toInt() < matJ.n_cols ) jlogresponses += matJ.col( jpredr.toInt() );
                      arma::vec jresponses = arma::exp( jlogresponses );
                      double jnorm = arma::accu( jresponses );  // 0.0;                                           // join normalization term (denominator)

                      // Replace overflowing distribs by max...
                      if( jnorm == 1.0/0.0 ) {
                        uint ind_max=0; for( int i=0; i<jlogresponses.size(); i++ ) if( jlogresponses(i)>jlogresponses(ind_max) ) ind_max=i;
                        jlogresponses -= jlogresponses( ind_max );
                        jresponses = arma::exp( jlogresponses );
                        jnorm = arma::accu( jresponses ); //accumulate is sum over elements
                        //                    jresponses.fill( 0.0 );  jresponses( ind_max ) = 1.0;
                        //                    jnorm = 1.0;
                      } //closes if jnorm

                      // For each possible no-join or join decision, and operator decisions...
                      for( JResponse jresponse; jresponse<JResponse::getDomain().getSize(); ++jresponse ) {
                        if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(probFork) + log(jresponses[jresponse.toInt()]/jnorm) > beams[t].rbegin()->getProb() ) {
                          J    j   = jresponse.getJoin();
                          EVar e   = jresponse.getE();
                          O    opL = jresponse.getLOp();
                          O    opR = jresponse.getROp();
                          if( jresponse.toInt() >= int(jresponses.size()) ) cerr << "ERROR: unknown jresponse: " << jresponse << endl;
                          double probJoin = jresponses[jresponse.toInt()] / jnorm;
                          if ( VERBOSE>1 ) cout << "       J ... " << " : " << jresponse << " = " << probJoin << endl;

                          // For each possible apex category label...
                          APredictor apredictor = q_tdec1.calcApexTypeCondition( f, j, e_p_t, e, opL, aLchild );  // save apredictor for use in prob calc
                          if ( VERBOSE>1 ) cout << "         A " << apredictor << "..." << endl;
                          if ( modA.end()!=modA.find(apredictor) )
                            for ( auto& tpA : modA.find(apredictor)->second ) {
                              if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(probFork) + log(probJoin) + log(tpA.second) > beams[t].rbegin()->getProb() ) {

                                if ( VERBOSE>1 ) cout << "         A " << apredictor << " : " << tpA.first << " = " << tpA.second << endl;

                                // For each possible brink category label...
                                BPredictor bpredictor = q_tdec1.calcBrinkTypeCondition( f, j, e_p_t, e, opL, opR, tpA.first, aLchild );  // bpredictor for prob calc
                                if ( VERBOSE>1 ) cout << "          B " << bpredictor << "..." << endl;
                                if ( modB.end()!=modB.find(bpredictor) )
                                  for ( auto& tpB : modB.find(bpredictor)->second ) {
                                    if ( VERBOSE>1 ) cout << "          B " << bpredictor << " : " << tpB.first << " = " << tpB.second << endl;
                                    //                            lock_guard<mutex> guard( mutexBeam );
                                    if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second) > beams[t].rbegin()->getProb() ) {

                                      if( chrono::high_resolution_clock::now() > tpLastReport + chrono::minutes(1) ) {
                                        tpLastReport = chrono::high_resolution_clock::now();
                                        lock_guard<mutex> guard( mutexMLSList );
                                        cerr << "WORKER " << numt << ": SENT " << currline << " WORD " << t << " FROM " << be_tdec1.getHidd() << " PRED " << ektpr_p_t << " JRESP " << jresponse << " A " << tpA.first << " B " << tpB.first << endl;
                                      } //closes if chrono
                                      // Calculate probability and storestate and add to beam...
                                      StoreState ss( q_tdec1, f, j, e_p_t, e, opL, opR, tpA.first, tpB.first, aPretrm, aLchild );
                                      if( (t<lwSent.size() && ss.size()>0) || (t==lwSent.size() && ss.size()==0) ) {
                                        beams[t].tryAdd( HiddState( aPretrm, f,e_p_t,k_p_t, jresponse, ss, tAnt-t ), ProbBack<HiddState>( lgpr_tdec1 + log(nprob) + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second), be_tdec1 ) ); 
                                        if( VERBOSE>1 ) cout << "                send (" << be_tdec1.getHidd() << ") to (" << ss << ") with "
                                          << (lgpr_tdec1 + log(nprob) + log(probFork) + log(probJoin) + log(tpA.second) + log(tpB.second)) << endl;
                                      } //closes if ( (t<lwSent
                                    } //closes if beams[t]
                                  } //closes for tpB
                              } //closes if beams[t]
                            } //closes for tpA
                        } //closes if beams[t]
                      } //closes for jresponse
                    } //closes if modP.end()
                  } //closes for f in {0,1}
                } //closes if beams[t]
              } //closes for ektpr_p_t
            } //closes for tAnt (second antecedent loop)
          } //closes be_tdec1

          // Write output...
          if ( numThreads == 1 ) cerr << " (" << beams[t].size() << ")";
          if ( VERBOSE ) cout << beams[t] << endl;
          { lock_guard<mutex> guard( mutexMLSList ); 
            cerr << "WORKER " << numt << ": SENT " << currline << " WORD " << t << endl;	
            //cerr << "Worker" << numt << " reached end of word loop, going to next word..." << endl;
          } //closes lock_guard
        } //closes for w lwSent  
        if ( numThreads == 1 ) cerr << endl;
        if ( VERBOSE ) cout << "MLS" << endl;

        //DelimitedList<psX,pair<HiddState,ProbBack>,psLine,psX> mls;

        { lock_guard<mutex> guard( mutexMLSList );
          if( numThreads > 1 ) cerr << "Finished line " << currline << " (" << beams[t].size() << ")..." << endl;
          cerr << "Worker: " << numt << " attempting to set mls on beams..." << endl;
          beams.setMostLikelySequence( mls );
          cerr << "length lbeWholeArticle: " << lbeWholeArticle.size() << endl;
          //lbeWholeArticle.insert(lbeWholeArticle.end(),mls.begin(),mls.end());
          mls.pop_back(); //remove dummy element before adding to lbe 
          lbeWholeArticle.insert(lbeWholeArticle.end(),mls.begin(),mls.end()); //insert mls at end of lbe
          cerr << "length lbeWholeArticle after insertion: " << lbeWholeArticle.size() << endl;
          //iterate over lbeWholeArticle, having each item backpoint to the previous
          for (auto it = lbeWholeArticle.begin(); it != lbeWholeArticle.end(); it++) {
            if ( it != lbeWholeArticle.begin() ) {
              it->setBack(*prev(it));
            }
          }
          cerr << "lbeWholeArticle.back().getBack().getHidd(): " << lbeWholeArticle.back().getBack().getHidd() << endl;
        }
      } //close loop lwSent over sents

      //{ lock_guard<mutex> guard( mutexMLSList ); 
      //  cerr << "Worker:" << numt << " processed all sents in an article, attempting to print..." << endl;
      //  cerr << "Worker: " << numt << " articles().size(): " << articles.size() << endl;
      //  cerr << "Worker: " << numt << " articles.front().size(): " << articles.front().size() << endl;
      //}

      { lock_guard<mutex> guard( mutexMLSList );
        //finished sent, now looping over global data and see whether it's ready to print
        //see if articles is not empty and first article is not empty and first sentence of first article is done, then print it.
        //cerr << "Worker: " << numt << " entered print loop" << endl;
        //cerr << "Worker: " << numt << " articleMLSs.size(): " << articleMLSs.size() << endl;
        //cerr << "Worker: " << numt << " articleMLSs.front().size(): " << articleMLSs.front().size() << endl;
        //cerr << "Worker: " << numt << " articles.front().size(): " << articles.front().size() << endl;
        while( articleMLSs.size()>0 && articleMLSs.front().size()>0 && articleMLSs.front().size()==articles.front().size() ) { 
          //cerr << "Worker: " << numt << " assigning mls for printing... " << endl;
          //auto& mls = articleMLSs.front().front(); //mls for first sentence of first article
          //cerr << "Worker: " << numt << " assigning sent for printing... " << endl;
          //auto& sent = articles.front().front(); // wordlist for first sentence of first article
          int u=1; 
          //cerr << "Worker: " << numt << " assigning iterator for beam element... " << endl;
          //auto ibe=next(mls.begin());  //iterator over beam elements?
          auto ibe=next(articleMLSs.front().front().begin());  //iterator over beam elements?
          //cerr << "Worker: " << numt << " assigning iterator for words... " << endl;
          //auto iw=sent.begin() ; //iterator over words
          auto iw=articles.front().front().begin() ; //iterator over words
          //cerr << "Worker: " << numt << " starting sentence print loop... " << endl;
          //for( ; (ibe != mls.end()) && (iw != sent.end()); ibe++, iw++, u++ ) {
          for( ; (ibe != articleMLSs.front().front().end()) && (iw != articles.front().front().end()); ibe++, iw++, u++ ) {
            cout << *iw << " " << ibe->getHidd() << " " << ibe->getProb(); //tokdecs output is: WORD HIDDSTATE PROB
            cout << endl;
          } //closes for ibe!=mls.end
          //cerr << "Worker: " << numt << " starting articleMLSs pop... " << endl;
          articleMLSs.front().pop_front(); //pop (mls of) first sentence of first article
          //cerr << "Worker: " << numt << " starting articles pop... " << endl;
          //cerr << "articles.front().size() before pop: " << articles.front().size() << endl;
          articles.front().pop_front(); //pop first sentence of first article
          //cerr << "articles.front().size() after pop: " << articles.front().size() << endl;
          //cerr << "Worker: " << numt << " testing articles.front().size()... " << endl;
          if (articles.front().size() == 0) {  //if article is empty then pop article
            //cerr << "Worker: " << numt << " first article is empty, attempting to pop articleMLSs... " << endl;
            articleMLSs.pop_front(); 
            //cerr << "Worker: " << numt << " first article is empty, attempting to pop articles... " << endl;
            articles.pop_front();
            //cerr << "popped front of articles" << endl;
          } 
        } //closes while articleMLSs 
      }//closes lock guard for print loop  
    } //closes while(True)
  }, numtglobal )); //brace closes for numtglobal

  for( auto& w : vtWorkers ) w.join();

} //closes int main

