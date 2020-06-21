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
#ifdef DENSE_VECTORS
#include <SemProcModels_dense.hpp>
#elif defined MLP
#include <SemProcModels_mlp.hpp>
#else
#include <SemProcModels_sparse.hpp>
#endif
#include <Beam.hpp>
int COREF_WINDOW = INT_MAX;
bool ABLATE_UNARY = false;
bool NO_ENTITY_BLOCKING = false;
bool NO_ANTUNK = false;

#define SERIAL_IO

////////////////////////////////////////////////////////////////////////////////

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";
const W wEmpty = W("");

////////////////////////////////////////////////////////////////////////////////

uint BEAM_WIDTH      = 1000;
uint VERBOSE         = 0;
uint OUTPUT_MEASURES = true;

////////////////////////////////////////////////////////////////////////////////

class Trellis : public vector<Beam<HiddState>> {
  public:
    Trellis ( ) : vector<Beam<HiddState>>() { reserve(100); }
    Beam<HiddState>& operator[] ( uint i ) { if ( i==size() ) emplace_back(BEAM_WIDTH); return vector<Beam<HiddState>>::operator[](i); }
    void setMostLikelySequence ( DelimitedList<psX,BeamElement<HiddState>,psLine,psX>& lbe, const JModel& jm ) {
      static StoreState ssLongFail( cFail, cFail );
//      static StoreState ssLongFail;  ssLongFail.emplace( ssLongFail.end() );  ssLongFail.back().apex().emplace_back(hvBot,cFail,S_A);  ssLongFail.back().base().emplace_back(hvBot,cFail,S_B);
      lbe.clear(); if( back().size()>0 ) lbe.push_front( *back().begin() );
      if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( lbe.front().getBack() );
      if( lbe.size()>0 ) lbe.emplace_back( BeamElement<HiddState>() );
      cerr << "lbe.size(): " << lbe.size() << endl;
      // If parse fails...
      if( lbe.size()==0 ) {
        cerr << "parse failed (lbe.size() = 0) " << "trellis size(): " << size() << endl;
        // Print a right branching structure...
        for( int t=size()-2; t>=0; t-- ) { 
          lbe.push_front( BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 1, EVar::eNil, K::kBot, jm.getResponse1(), ssLongFail ) ) ); // fork and join
        }
//        cerr << "size of lbe after push_fronts: " << lbe.size() << endl;
        lbe.front() = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 1, EVar::eNil, K::kBot, jm.getResponse0(), ssLongFail ) );       // front: fork no-join
        lbe.back( ) = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 0, EVar::eNil, K::kBot, jm.getResponse1(), StoreState() ) );     // back: join no-fork
//        cerr << "size of lbe after front and back assignments: " << lbe.size() << endl;
        if( size()==2 ) {  //special case if single word, fork and join
//          cerr << "assigning front of fail lbe" << endl;
          lbe.front() = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 1, EVar::eNil, K::kBot, jm.getResponse1(), StoreState() ) );   // unary case: fork and join
        }
        // Add dummy element (not sure why this is needed)...
        lbe.push_front( BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 0, EVar::eNil, K::kBot, jm.getResponse0(), StoreState() ) ) ); // no-fork, no-join?
        //start experiment - next two lines switch front element to nofork,join, add additional dummy at rear
        //TODO to revert, comment out next two, comment in pushfront above
        lbe.emplace_back( BeamElement<HiddState>() );
        //end epxeriment

//        cerr << "size of lbe after dummy push_front: " << lbe.size() << endl;
        cerr<<"parse failed"<<endl;
        // does lbe here consist of a single sentence or of the whole article?
      }
      // For each element of MLE after first dummy element...
      //for ( auto& be : lbe ) { cerr << "beam element hidd: " << be.getHidd() << endl; } //TODO confirm includes all words, count initial/final dummies
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

const BeamElement<HiddState>* getNextAntecedent (const BeamElement<HiddState>* antPtr) {
    //updates antPtr to point to next antecedent back, using i index to count back that many words/timesteps
    int i = antPtr->getHidd().getI(); //offset to antecedent timestep
    i = abs(i);
    for (; i != 0; i--) {
      antPtr = &antPtr->getBack();
    }
    return antPtr;
  }

W getHistWord ( const BeamElement<HiddState>* antPtr, const W wEmpty, bool NO_ANTUNK ) {
  //cout << "getHistWord started with antPtr: " << antPtr->getHidd() << " located at: " << antPtr << " with original k: " << antPtr->getHidd().getForkK() << endl;
  //W histword = W(""); //default case - no unk, no coreference, histword is "" POSSIBLY NOT THREAD SAFE
  W histword = wEmpty; //default case - no unk, no coreference, histword is ""
  if (NO_ANTUNK) return histword;
  if (antPtr->getHidd().getForkK().isUnk()) { //unk case - unk, histword is word, short-circuits inheritance case
    histword = antPtr->getHidd().getWord();
    return histword;  
  } 
  for ( ; (antPtr->getHidd().getI() != 0) ; antPtr = getNextAntecedent(antPtr) ) { //inheritance case - break upon finding most recent unk in antecedent chain, word is that unk's word
    //cout << "getHistWord looping, at antPtr's hidd: " << antPtr->getHidd() << " located at: " << antPtr << "with k: " << antPtr->getHidd().getForkK() << endl;
    if (antPtr->getHidd().getForkK().isUnk()) { 
      histword = antPtr->getHidd().getWord(); //at most recent unk, get observed word and return
      break;
    }
  }
  return histword;
}

  
////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  uint numThreads = 1;

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

  { // Define model lists...
    list<DelimitedTrip<psX,WPredictor,psSpcColonSpc,W,psSpcEqualsSpc,Delimited<double>,psX>> lW;

    // For each command-line flag or model file...
    for ( int a=1; a<nArgs; a++ ) {
      if      ( 0==strcmp(argv[a],"-v") ) VERBOSE = 1;
      else if ( 0==strcmp(argv[a],"-V") ) VERBOSE = 2;
      else if ( 0==strncmp(argv[a],"-p",2) ) numThreads = atoi(argv[a]+2);
      else if ( 0==strncmp(argv[a],"-b",2) ) BEAM_WIDTH = atoi(argv[a]+2);
//      else if ( 0==strcmp(argv[a],"-c") ) OUTPUT_MEASURES = 1;
      else if ( 0==strncmp(argv[a],"-f",2) ) FEATCONFIG = atoi(argv[a]+2);
      else if( '-'==argv[a][0] && 'c'==argv[a][1] && '\0'!=argv[a][2] ) COREF_WINDOW = atoi( argv[a]+2 );
      //else if ( string(argv[a]) == "t" ) STORESTATE_TYPE = true;
      else if( '-'==argv[a][0] && 'a'==argv[a][1] ) ABLATE_UNARY = true;
      else if( '-'==argv[a][0] && 'n'==argv[a][1] && 'b'==argv[a][2]) NO_ENTITY_BLOCKING = true;
      else if( '-'==argv[a][0] && 'n'==argv[a][1] && 'a'==argv[a][2]) NO_ANTUNK = true;
      else {
        cerr << "Loading model " << argv[a] << "..." << endl;
        // Open file...
        ifstream fin (argv[a], ios::in );
        // Read model lists...
        int linenum = 0;
        while ( fin && EOF!=fin.peek() ) {
          if      ( fin.peek()=='E' ) matEmutable = EMat( fin );
          else if ( fin.peek()=='O' ) funcOmutable = OFunc( fin );
          else if ( fin.peek()=='N' ) modNmutable = NModel( fin );
          else if ( fin.peek()=='F' ) modFmutable = FModel( fin );
          else if ( fin.peek()=='P' ) modPmutable = PModel( fin );
          else if ( fin.peek()=='W' ) modWmutable = WModel( fin );  //fin >> "W " >> *lW.emplace(lW.end()) >> "\n";
          else if ( fin.peek()=='J' ) modJmutable = JModel( fin );
          else if ( fin.peek()=='A' ) modAmutable = AModel( fin );
          else if ( fin.peek()=='B' ) modBmutable = BModel( fin );
          else {
            Delimited<string> sOffSpec;
            fin >> sOffSpec >> "\n";
            cerr << "WARNING: skipping off-spec input line: '" << sOffSpec << "'" << endl;
          } 
          if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
        }
        cerr << "Model " << argv[a] << " loaded." << endl;
      }
    } //closes for int a=1

//    // Populate model structures...
//    for ( auto& prw : lW ) lexW[prw.second()].emplace_back(prw.first(),prw.third());
  } //closes define model lists

//  modJmutable.getResponseIndex( 0, EVar::eNil, O_N, O_I );
//  modJmutable.getResponseIndex( 1, EVar::eNil, O_N, O_I );

  const EMat&   matE  = matEmutable;
  const OFunc&  funcO = funcOmutable;
  const NModel& modN  = modNmutable;
  const FModel& modF  = modFmutable;
  const PModel& modP  = modPmutable;
  const WModel& modW  = modWmutable;
  const JModel& modJ  = modJmutable;
  const AModel& modA  = modAmutable;
  const BModel& modB  = modBmutable;

//  const vec& test = arma::ones<vec>(40);
//  const vec& testf = arma::ones<vec>(7+SYN_SIZE+2*SEM_SIZE);
//  const vec& testj = arma::ones<vec>(7+2*SYN_SIZE+3*SEM_SIZE);
  cerr<<"Models ready."<<endl;
//  cerr << "Experiments with 1s" << endl;
//  cerr << "OFunc(1) " << funcO(1, test) << endl;
//  cerr << "OFunc(-1) " << funcO(-1, test) << endl;
//  cerr << "OFunc(2) " << funcO(2, test) << endl;
//  cerr << "OFunc(-2) " << funcO(-2, test) << endl;
//  cerr << "OFunc(3) " << funcO(3, test) << endl;
//  cerr << "OFunc(-3) " << funcO(-3, test) << endl;
//  cerr << "OFunc(4) " << funcO(4, test) << endl;
//  cerr << "OFunc(-4) " << funcO(-4, test) << endl;
//  cerr << "F " << modF.testCalcResponses(testf) << endl;
//  cerr << "J " << modJ.testCalcResponses(testj) << endl;

  mutex mutexMLSList;
  vector<thread> vtWorkers;  vtWorkers.reserve( numThreads );

  if( OUTPUT_MEASURES ) cout << "word pos f j store ndec totsurp" << endl;

#ifdef SERIAL_IO
  // List of articles, which are pairs of lists of lists of words and lists of lists of hidd states...
  list< pair< DelimitedList<psX,DelimitedList<psX,ObsWord,psSpace,psX>,psLine,psX>, DelimitedList<psX,DelimitedList<psX,BeamElement<HiddState>,psLine,psX>,psLine,psX> > > corpus;
  // Read in list of articles...
//  while( cin.peek() != EOF ) { cin >> corpus.emplace( corpus.end() )->first >> "!ARTICLE\n";  cerr<<"i got: "<<corpus.back().first.size()<<endl;  }
  while( cin.peek() != EOF ) {
    if( cin.peek() == '!' ) cin >> "!ARTICLE\n";
    corpus.emplace_back();
    while( cin.peek() != '!' && cin.peek() != EOF )  cin >> *corpus.back().first.emplace( corpus.back().first.end() ) >> "\n";
    cerr<<"I read an article with " << corpus.back().first.size() << " sentences." << endl;
  }
  // Pointers to 
  auto iartNextToProc = corpus.begin();
  auto iartNextToDump = corpus.begin();
#else
  uint linenum = 0;
  // For each line in stdin...
  list<list<DelimitedList<psX,ObsWord,psSpace,psX>>> articles; //list of list of sents. each list of sents is an article.
  list<list<DelimitedList<psX,BeamElement<HiddState>,psLine,psX>>> articleMLSs; //list of MLSs
#endif

  // loop over threads (each thread gets an article)
  for( uint numtglobal=0; numtglobal<numThreads; numtglobal++ ) vtWorkers.push_back( thread( [&corpus,&iartNextToProc,&iartNextToDump,/*&articleMLSs,&articles,&linenum,*/&mutexMLSList,numThreads,&matE,&funcO,&modN,&modF,&modP,&modW,&modJ,&modA,&modB] (uint numt) {

    auto tpLastReport = chrono::high_resolution_clock::now();  // clock for thread heartbeats

    // Loop over articles...
    while( true ) {

#ifdef SERIAL_IO
      decltype(corpus)::iterator iart;
      { lock_guard<mutex> guard( mutexMLSList );
        if( iartNextToProc == corpus.end() ) break;
        iart = iartNextToProc++;
      }
      const auto& sents = iart->first;
      auto&       MLSs  = iart->second;

      int currline = 0;
#else
      // Read in your worker thread's article in this lock block
      mutexMLSList.lock( );
      if( not ( cin && EOF!=cin.peek() ) ) { mutexMLSList.unlock(); break; }

      uint currline = linenum; 
      articles.emplace_back(); 
      auto& sents = articles.back(); //a specific article becomes the thread's sents //returns reference
      articleMLSs.emplace_back();
      auto& MLSs = articleMLSs.back();

      DelimitedList<psX,ObsWord,psSpace,psX> articleDelim; // !article should be consumed between sentence reads
      //loop over sentences in an article
      cin >> articleDelim >> "\n"; //consume !ARTICLE

      while (cin.peek()!='!' and cin.peek()!=EOF) {
        // Read sentence...
        linenum++; //updates linenum for when handing off to other thread
        DelimitedList<psX,ObsWord,psSpace,psX> lwSent; // init input list for each iteration - otherwise cin just keeps appending to existing lwSent
        cin >> lwSent >> "\n";
        sents.emplace_back( lwSent );
      }
      mutexMLSList.unlock();
#endif

      if ( numThreads == 1 ) cerr << "#" << currline;

      DelimitedList<psX,BeamElement<HiddState>,psLine,psX> lbeWholeArticle;
      lbeWholeArticle.emplace_back(); //create null beamElement at start of article

      for (auto& lwSent : sents) {
        currline++;

#ifdef SERIAL_IO
#else
//        mutexMLSList.lock();
        // Add mls to list...
        MLSs.emplace_back( ); //establish placeholder for mls for this specific sentence
        auto& mls = MLSs.back();
//        mutexMLSList.unlock(); 
#endif

        Trellis   beams;  // sequence of beams - each beam is hypotheses at a given timestep
        uint      t=0;    // time step

        // Allocate space in beams to avoid reallocation...
        // Create initial beam element...
        //TODO see if resetting each sentences to use zero prob instead of last prob avoids underflow
        lbeWholeArticle.back().setProb() = 0.0;
        beams[0].tryAdd( lbeWholeArticle.back().getHidd(), lbeWholeArticle.back().getProbBack() );
        //beams[0].tryAdd( HiddState(), ProbBack<HiddState>() );

        // For each word... 
        for ( auto& w_t : lwSent ) {
          try {
          if ( numThreads == 1 ) cerr << " " << w_t;
          if ( VERBOSE ) cout << "WORD:" << w_t << endl;

          // Create beam for current time step...
          beams[++t].clear();
          //if (w_t == W("``") and currline == 37) { VERBOSE += 1; } //corontowsj01 208first.158onward sent immediately after jim enzor sent crashes, outputting only first `` token.
          // For each hypothesized storestate at previous time step...
          for( const BeamElement<HiddState>& be_tdec1 : beams[t-1] ) { //beams[t-1] is a Beam<ProbBack,BeamElement>, so be_tdec1 is a beam item, which is a pair<ProbBack,BeamElement>. first.first is the prob in the probback, and second is the beamelement, which is a sextuple of <sign, f, e, k, j, q>
            double            lgpr_tdec1 = be_tdec1.getProb(); // logprob of prev storestate
            const StoreState& q_tdec1    = be_tdec1.getHidd().sixth();  // prev storestate
            if( VERBOSE>1 ) cout << "  from (" << be_tdec1.getHidd() << ")" << endl;
            const ProbBack<HiddState> pbDummy = ProbBack<HiddState>(0.0, be_tdec1); //dummy element for most recent timestep
            const HiddState hsDummy = HiddState(Sign(/*hvTop,CVar(),S()*/),F(),EVar(),K(),JResponse(),StoreState(),0 ); //dummy hidden state with kTop semantics 
            const BeamElement<HiddState> beDummy = BeamElement<HiddState>(pbDummy, hsDummy); //at timestep t, represents null antecedent 
            const BeamElement<HiddState>* pbeAnt = &beDummy;
            //calculate denominator / normalizing constant over all antecedent timesteps
            double ndenom = 0.0;
            vector<int> excludedIndices; //initialize blocking list

            if (VERBOSE > 1) cout << "entering denom loop... " << &pbeAnt->getBack() << endl;
            //denominator loop over candidate antecedents
            for ( int tAnt = t; (&pbeAnt->getBack() != &BeamElement<HiddState>::beStableDummy) && (int(t-tAnt)<=COREF_WINDOW); tAnt--, pbeAnt = &pbeAnt->getBack()) { 
              if (VERBOSE > 1) cout << "entered denom loop... " << &pbeAnt->getBack() << endl;
              if ( pbeAnt->getHidd().getI() != 0 ) {
                if (VERBOSE > 1) cout << "    adding index to exclude for blocking: " << tAnt+pbeAnt->getHidd().getI() << " pbeAnt...get(): " << pbeAnt->getHidd().getI() << endl;
                excludedIndices.push_back(tAnt+pbeAnt->getHidd().getI()); //add excluded index if there's a non-null coref decision
              }
              if (NO_ENTITY_BLOCKING == false) {
                if (std::find(excludedIndices.begin(), excludedIndices.end(), tAnt) != excludedIndices.end()){
                  continue; //skip excluded indices
                }
              }
              bool corefON = (tAnt==int(t)) ? 0 : 1;

              if (VERBOSE > 1) cout << "about to generate npv... " << &pbeAnt->getBack() << endl;
              NPredictorVec npv( modN, pbeAnt->getHidd().getPrtrm(), corefON, t - tAnt, q_tdec1, ABLATE_UNARY );
              if (VERBOSE > 1) cout << "about to generate nresponses... " << &pbeAnt->getBack() << endl;
              arma::vec nresponses = modN.calcResponses( npv ); //nps.NLogResponses(matN);

              //if (VERBOSE > 1) {
              //  cout << "added 1 hypoth to ndenom with score for 1: " << nlogresponses(1) << " score for 0: " << nlogresponses(0) << endl;
              //  cout << "    basec: " << npv.getBaseC() << " antec: " << npv.getAnteC() << " basesem: " << npv.getBaseSem() << " antesem: " << npv.getAnteSem() << " antdist: " << npv.getAntDist() << "sqantdist: " << npv.getAntDistSq() << " corefOn: " << npv.getCorefOn() << endl;
              //}
              
              ndenom += nresponses(1) / nresponses(0) ;
              if (VERBOSE > 1) cout << "bottom of denom loop... " << &pbeAnt->getBack() << endl;
            } //closes for tAnt

            pbeAnt = &beDummy; //reset pbiAnt pointer after calculating denominator

            if (VERBOSE > 1) cout << "entering numerator loop..." << endl;
            //numerator loop over candidate antecedents. specific choice.
            for ( int tAnt = t; (&pbeAnt->getBack() != &BeamElement<HiddState>::beStableDummy) && (int(t-tAnt)<=COREF_WINDOW); tAnt--, pbeAnt = &pbeAnt->getBack()) { //numerator, iterate over candidate antecedent ks, following trellis backpointers. 
              //block indices as read from previous storestate's excludedIndices
              if (std::find(excludedIndices.begin(), excludedIndices.end(), tAnt) != excludedIndices.end()){
                continue;
              }
              
              const HVec& hvAnt = pbeAnt->getHidd().getPrtrm().getHVec();

              //Calculate antecedent N model predictors 
              bool corefON = (tAnt==int(t)) ? 0 : 1;
              NPredictorVec npv( modN, pbeAnt->getHidd().getPrtrm(), corefON, t - tAnt, q_tdec1, ABLATE_UNARY );
              //if (VERBOSE>1) { cout << "    " << pair<const NModel&, const NPredictorVec&>(modN,npv) << endl; } //npv.printOut(cout); }
              arma::vec nresponses = modN.calcResponses( npv );
              if (VERBOSE>1) {
                cout << "considering basec: " << npv.getBaseC() << " antec: " << npv.getAnteC() << " basesem: " << npv.getBaseSem() << " antesem: " << npv.getAnteSem() << " antdist: " << npv.getAntDist() << " sqantdist: " << npv.getAntDistSq() << " corefOn: " << npv.getCorefOn() << endl;
                cout << "  with nresponses: " << nresponses << " nresponses(1): " << nresponses(1) << " nresponses(0): " << nresponses(0) << endl;
                cout << "  nr(1) / nr(0): " << nresponses(1) / nresponses(0) << endl;
              }
              double numerator = nresponses(1) / nresponses(0) ;
              double nprob = numerator / ndenom;

              if ( VERBOSE>1 ) cout << "    N ... : 1 = " << numerator << "/" << ndenom << "=" << nprob << "  tAnt: " << (t - tAnt) << endl;

              if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) > beams[t].rbegin()->getProb() ) {
                FPredictorVec lfpredictors( modF, hvAnt, not corefON, q_tdec1 );
//                if( VERBOSE>1 ) cout << "     f predictors: " << pair<const FModel&,const FPredictorVec&>( modF, lfpredictors ) << endl;
                arma::vec fresponses = modF.calcResponses( lfpredictors );
              
                //get most recent observed word for which k of fek F decision was 'unk'.
                const BeamElement<HiddState>* antPtr = pbeAnt; 
                //cout << "main semproc got ptr loc: " << antPtr << endl;
                W histword;
                histword = getHistWord(antPtr, wEmpty, NO_ANTUNK);
                //cout << "main semproc got histword: " << histword << endl;

                // For each possible lemma (context + label + prob) for preterminal of current word...
                for ( auto& ektpr_p_t : modW.calcPredictorLikelihoods(w_t, histword) ) { //ektpr_p_t is a pair of (Wpredictor, prob)
                  if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(ektpr_p_t.second) > beams[t].rbegin()->getProb() ) {
                    EVar  e_p_t       = ektpr_p_t.first.first();
                    K     k_p_t       = (FEATCONFIG & 8 && ektpr_p_t.first.second().getString()[2]!='y') ? K::kBot : ektpr_p_t.first.second();   // context of current preterminal
                    CVar  c_p_t       = ektpr_p_t.first.third();                               // label of current preterminal
                    double probwgivkl = ektpr_p_t.second;                                     // probability of current word given current preterminal

                    if ( VERBOSE>1 ) cout << "     W " << e_p_t << " " << k_p_t << " " << c_p_t << " : " << w_t << " = " << probwgivkl << endl;

                    // For each possible no-fork or fork decision...
                    for ( auto& f : {0,1} ) if ( q_tdec1.size() > 0 or f > 0 ) {
//                      if( modF.getResponseIndex(f,e_p_t,k_p_t)==0 ) cerr<<"ERROR: unable to find fresponse "<<f<<"&"<<e_p_t<<"&"<<k_p_t<<endl;
                      if( modF.getResponseIndex(f,e_p_t,k_p_t) == uint(-1) ) continue;
                      double probFork = fresponses( modF.getResponseIndex(f,e_p_t,k_p_t) );
                      if ( VERBOSE>1 ) cout << "      F ... : " << f << " " << e_p_t << " " << k_p_t << " = " << probFork << endl;

                      // Thread heartbeat (to diagnose thread death)...
                      if( chrono::high_resolution_clock::now() > tpLastReport + chrono::minutes(1) ) {
                        tpLastReport = chrono::high_resolution_clock::now();
                        lock_guard<mutex> guard( mutexMLSList );
                        cerr << "WORKER " << numt << ": SENT " << currline << " WORD " << t << " FROM " << be_tdec1.getHidd() << " PRED " << ektpr_p_t << endl;
                      } //closes if chrono

                      // If preterminal prob is nonzero...
                      PPredictorVec ppredictor( f, e_p_t, k_p_t, q_tdec1 );
                      if ( VERBOSE>1 ) cout << "       P " << ppredictor << " : " << c_p_t << "...?" << endl;
                      if ( modP.end()!=modP.find(ppredictor) && modP.find(ppredictor)->second.end()!=modP.find(ppredictor)->second.find(c_p_t) ) {

                        if ( VERBOSE>1 ) cout << "       P " << ppredictor << " : " << c_p_t << " = " << modP.find(ppredictor)->second.find(c_p_t)->second << endl;

                        // Calc probability for fork phase...
                        double probFPW = probFork * modP.find(ppredictor)->second.find(c_p_t)->second * probwgivkl;
                        if ( VERBOSE>1 ) cout << "       f: f" << f << "&" << e_p_t << "&" << k_p_t << " " << probFork << " * " << modP.find(ppredictor)->second.find(c_p_t)->second << " * " << probwgivkl << " = " << probFPW << endl;

                        StoreState qPretrm( q_tdec1, hvAnt, e_p_t, k_p_t, c_p_t, matE, funcO );
                        const Sign& aPretrm = qPretrm.getApex();
                        if( VERBOSE>1 ) cout << "       qPretrm="    << qPretrm    << endl;
                        StoreState qTermPhase( qPretrm, f );
                        const Sign& aLchild = qTermPhase.getApex();
                        if( VERBOSE>1 ) cout << "       qTermPhase=" << qTermPhase << endl;

                        JPredictorVec ljpredictors( modJ, f, e_p_t, aLchild, qTermPhase );  // q_tdec1.calcJoinPredictors( ljpredictors, f, e_p_t, aLchild, false ); // predictors for join
//                        if( VERBOSE>1 ) cout << "        j predictors: " << pair<const JModel&,const JPredictorVec&>( modJ, ljpredictors ) << endl;
                        arma::vec jresponses = modJ.calcResponses( ljpredictors );

                        // For each possible no-join or join decision, and operator decisions...
                        for( unsigned int jresponse=0; jresponse<jresponses.size(); jresponse++ ) {  //JResponse jresponse; jresponse<JResponse::getDomain().getSize(); ++jresponse ) {
                          if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(probFPW) + log(jresponses[jresponse]/* /jnorm */) > beams[t].rbegin()->getProb() ) {
                            J    j   = modJ.getJEOO( jresponse ).first();  //.getJoin();
                            EVar e   = modJ.getJEOO( jresponse ).second(); //.getE();
                            O    opL = modJ.getJEOO( jresponse ).third();  //.getLOp();
                            O    opR = modJ.getJEOO( jresponse ).fourth(); //.getROp();
                            //if( jresponse.toInt() >= int(jresponses.size()) ) cerr << "ERROR: unknown jresponse: " << jresponse << endl;
                            double probJoin = jresponses[jresponse]; //  / jnorm;
                            if ( VERBOSE>1 ) cout << "        J ... " << " : " << modJ.getJEOO(jresponse) << " = " << probJoin << endl;

                            // For each possible apex category label...
                            APredictorVec apredictor( f, j, e_p_t, e, opL, aLchild, qTermPhase );  // save apredictor for use in prob calc
                            if ( VERBOSE>1 ) cout << "         A " << apredictor << "..." << endl;
                            if ( modA.end()!=modA.find(apredictor) )
                              for ( auto& cpA : modA.find(apredictor)->second ) {
                                if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(probFPW) + log(probJoin) + log(cpA.second) > beams[t].rbegin()->getProb() ) {

                                  if ( VERBOSE>1 ) cout << "         A " << apredictor << " : " << cpA.first << " = " << cpA.second << endl;

                                  // For each possible brink category label...
                                  BPredictorVec bpredictor( f, j, e_p_t, e, opL, opR, cpA.first, aLchild, qTermPhase );  // bpredictor for prob calc
                                  if ( VERBOSE>1 ) cout << "          B " << bpredictor << "..." << endl;
                                  if ( modB.end()!=modB.find(bpredictor) )
                                    for ( auto& cpB : modB.find(bpredictor)->second ) {
                                      if ( VERBOSE>1 ) cout << "          B " << bpredictor << " : " << cpB.first << " = " << cpB.second << endl;
                                      //                            lock_guard<mutex> guard( mutexBeam );
                                      if( beams[t].size()<BEAM_WIDTH || lgpr_tdec1 + log(nprob) + log(probFPW) + log(probJoin) + log(cpA.second) + log(cpB.second) > beams[t].rbegin()->getProb() ) {

                                        // Thread heartbeat (to diagnose thread death)...
                                        if( chrono::high_resolution_clock::now() > tpLastReport + chrono::minutes(1) ) {
                                          tpLastReport = chrono::high_resolution_clock::now();
                                          lock_guard<mutex> guard( mutexMLSList );
                                          cerr << "WORKER " << numt << ": SENT " << currline << " WORD " << t << " FROM " << be_tdec1.getHidd() << " PRED " << ektpr_p_t << " JRESP " << modJ.getJEOO(jresponse) << " A " << cpA.first << " B " << cpB.first << endl;
                                        } //closes if chrono
                                        // Calculate probability and storestate and add to beam...
                                        StoreState ss( qTermPhase, j, e, opL, opR, cpA.first, cpB.first );
                                        if( (t<lwSent.size() && ss.size()>0) || (t==lwSent.size() && ss.size()==0) ) {
                                          beams[t].tryAdd( HiddState( aPretrm, f,e_p_t,k_p_t, jresponse, ss, tAnt-t, w_t ), ProbBack<HiddState>( lgpr_tdec1 + log(nprob) + log(probFPW) + log(probJoin) + log(cpA.second) + log(cpB.second), be_tdec1 ) ); 
                                          if( VERBOSE>1 ) cout << "                send (" << be_tdec1.getHidd() << ") to (" << ss << ") with "
                                            << (lgpr_tdec1 + log(nprob) + log(probFPW) + log(probJoin) + log(cpA.second) + log(cpB.second)) << endl;
                                        } //closes if ( (t<lwSent
                                      } //closes if beams[t]
                                    } //closes for cpB
                                } //closes if beams[t]
                              } //closes for cpA
                          } //closes if beams[t]
                        } //closes for jresponse
                      } //closes if modP.end()
                    } //closes for f in {0,1}
                  } //closes if beams[t]
                } //closes for ektpr_p_t
              } // short-circuit bad nprob
            } //closes for tAnt (second antecedent loop)
          } //closes be_tdec1

          // Write output...
          if ( numThreads == 1 ) cerr << " (" << beams[t].size() << ")";
          if ( VERBOSE ) { //cout << beams[t] << endl;
            cout << "BEAM" << endl;
            for( auto& be : beams[t] )
              cout << be.getProb() << " " << be.getHidd().first() << " f" << be.getHidd().second() << "&" << be.getHidd().third() << "&" << be.getHidd().fourth() << " j" << modJ.getJEOO(be.getHidd().fifth()) << " " << be.getHidd().sixth() << " " << be.getHidd().seventh() << " me: " << &be << " myback: " << &be.getBack() << endl; //tokdecs output is: WORD HIDDSTATE PROB
          }
          { lock_guard<mutex> guard( mutexMLSList ); 
            cerr << "WORKER " << numt << ": SENT " << currline << " WORD " << t << endl;	
          } //closes lock_guard
          } catch(int e) {
            cerr << "caught error, crash imminent..." << endl;
            cerr << "WORKER " << numt << ": SENT " << currline << " WORD " << t << endl;
            for (auto w : lwSent) cerr << w << " ";
            cerr << endl;
          }//closes try
        } //closes for w lwSent  
        if ( numThreads == 1 ) cerr << endl;
        if ( VERBOSE ) cout << "MLS" << endl;

        //DelimitedList<psX,pair<HiddState,ProbBack>,psLine,psX> mls;

        { lock_guard<mutex> guard( mutexMLSList );
#ifdef SERIAL_IO
          auto& mls = *MLSs.emplace( MLSs.end() ); //establish placeholder for mls for this specific sentence
          //auto& mls = MLSs.back();
#else
#endif 
          if( numThreads > 1 ) cerr << "Finished line " << currline << " (" << beams[t].size() << ")..." << endl;
          //cerr << "Worker: " << numt << " attempting to set mls on beams..." << endl;
          beams.setMostLikelySequence( mls, modJ );
          //cerr << "length lbeWholeArticle: " << lbeWholeArticle.size() << endl;
          mls.pop_back(); //remove dummy element before adding to lbe 
          lbeWholeArticle.insert(lbeWholeArticle.end(), next(mls.begin()), mls.end()); //insert mls at end of lbe
          //cerr << "length lbeWholeArticle after insertion: " << lbeWholeArticle.size() << endl;
          //iterate over lbeWholeArticle, having each item backpoint to the previous
          for (auto it = lbeWholeArticle.begin(); it != lbeWholeArticle.end(); it++) {
            if ( it != lbeWholeArticle.begin() ) {
              it->setBack(*prev(it));
            }
          }
          //cerr << "lbeWholeArticle.back().getBack().getHidd(): " << lbeWholeArticle.back().getBack().getHidd() << endl;
        }
      } //close loop lwSent over sents

      { lock_guard<mutex> guard( mutexMLSList );
        //cerr << "concbug: checking to print..." << endl;
        //finished sent, now looping over global data and see whether it's ready to print
        //see if articles is not empty and first article is not empty and first sentence of first article is done, then print it.
#ifdef SERIAL_IO
        while( iartNextToDump != corpus.end() and iartNextToDump->first.size() == iartNextToDump->second.size() ) {
//          for( auto& mls : (iartNextToDump++)->second ) {
//            for( auto& be : mls ) {
//              cout << be.getHidd() << endl;
////          cout << (iartNextToDump++)->second << endl;
          cout << "!ARTICLE" << endl;
          auto isent = iartNextToDump->first.begin();  // Iterator over sentences.
          auto imls  = iartNextToDump->second.begin(); // Iterator over most likely sequences.
          for( ;  isent != iartNextToDump->first.end() and imls != iartNextToDump->second.end();  isent++, imls++ ) {
            auto iw  = isent->begin();         // Iterator over words.
            auto ibe = next( imls->begin() );  // Iterator over mls elements.
            for( ;  iw != isent->end() and ibe != imls->end();  ibe++, iw++ )
//              cerr << "trying to dump id=" << ibe->getHidd().fifth() << " EVar=" << modJ.getJEOO(ibe->getHidd().fifth()).second().toInt() << endl;
              cout << *iw << " " << ibe->getHidd().first() << " f" << ibe->getHidd().second() << "&" << ibe->getHidd().third() << "&" << ibe->getHidd().fourth() << " j" << modJ.getJEOO(ibe->getHidd().fifth()) << " " << ibe->getHidd().sixth() << " " << ibe->getHidd().seventh() << " " << ibe->getProb() << endl; //tokdecs output is: WORD HIDDSTATE PROB
          }
          iartNextToDump++;
        }
#else
        while( articleMLSs.size()>0 && articleMLSs.front().size()>0 && articleMLSs.front().size()==articles.front().size() ) { 
          int u=1; 
          auto ibe=next(articleMLSs.front().front().begin());  //iterator over beam elements?
          auto iw=articles.front().front().begin() ; //iterator over words
          for( ; (ibe != articleMLSs.front().front().end()) && (iw != articles.front().front().end()); ibe++, iw++, u++ ) {
//            cerr << "trying to dump id=" << ibe->getHidd().fifth() << " EVar=" << modJ.getJEOO(ibe->getHidd().fifth()).second().toInt() << endl;
            cout << *iw << " " << ibe->getHidd().first() << " f" << ibe->getHidd().second() << "&" << ibe->getHidd().third() << "&" << ibe->getHidd().fourth() << " j" << modJ.getJEOO(ibe->getHidd().fifth()) << " " << ibe->getHidd().sixth() << " " << ibe->getHidd().seventh() << " " << ibe->getProb(); //tokdecs output is: WORD HIDDSTATE PROB
            cout << endl;
          } //closes for ibe!=mls.end
          articleMLSs.front().pop_front(); //pop (mls of) first sentence of first article
          articles.front().pop_front(); //pop first sentence of first article
          if (articles.front().size() == 0) {  //if article is empty then pop article
            articleMLSs.pop_front(); 
            articles.pop_front();
          } 
        } //closes while articleMLSs 
        //cerr << "concbug: done checking to print." << endl;
#endif
      } //closes lock guard for print loop  
    } //closes while(True)
  }, numtglobal )); //brace closes for numtglobal

  for( auto& w : vtWorkers ) w.join();

} //closes int main

