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

uint BEAM_WIDTH = 1000;

class Trellis : public vector<Beam<HiddState>> {
  public:
    Trellis ( ) : vector<Beam<HiddState>>() { reserve(100); }
    Beam<HiddState>& operator[] ( uint i ) { if ( i==size() ) emplace_back(BEAM_WIDTH); return vector<Beam<HiddState>>::operator[](i); }
    void setMostLikelySequence ( DelimitedList<psX,BeamElement<HiddState>,psLine,psX>& lbe, const JModel& jm ) {
      static StoreState ssLongFail( cFail, cFail );
//      static StoreState ssLongFail;  ssLongFail.emplace( ssLongFail.end() );  ssLongFail.back().apex().emplace_back(hvBot,cFail,S_A);  ssLongFail.back().base().emplace_back(hvBot,cFail,S_B);
      // Add top of last timestep beam to front of mls list...
      lbe.clear(); if( back().size()>0 ) lbe.push_front( *back().begin() );
      // Follow backpointers from trellis and add each to front of mls list...
      if( lbe.size()>0 ) for( int t=size()-2; t>=0; t-- ) lbe.push_front( lbe.front().getBack() );
      // Add dummy element at end...
      if( lbe.size()>0 ) lbe.emplace_back( BeamElement<HiddState>() );
      cerr << "lbe.size(): " << lbe.size() << endl;
      // If parse fails...
      if( lbe.size()==0 ) {
        cerr << "parse failed (lbe.size() = 0) " << "trellis size(): " << size() << endl;
        // Print a right branching structure...
        for( int t=size()-2; t>=0; t-- ) { 
          lbe.push_front( BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 1, EVar::eNil, K::kBot, jm.getResponse1(), ssLongFail, Sign(hvBot,cFail,S_A) ) ) ); // fork and join
        }
//        cerr << "size of lbe after push_fronts: " << lbe.size() << endl;
        lbe.front() = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 1, EVar::eNil, K::kBot, jm.getResponse0(), ssLongFail, Sign(hvBot,cFail,S_A) ) );       // front: fork no-join
        lbe.back( ) = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 0, EVar::eNil, K::kBot, jm.getResponse1(), StoreState(), Sign(hvBot,cFail,S_A) ) );     // back: join no-fork
//        cerr << "size of lbe after front and back assignments: " << lbe.size() << endl;
        if( size()==2 ) {  //special case if single word, fork and join
//          cerr << "assigning front of fail lbe" << endl;
          lbe.front() = BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 1, EVar::eNil, K::kBot, jm.getResponse1(), StoreState(), Sign(hvBot,cFail,S_A) ) );   // unary case: fork and join
        }
        // Add dummy element (not sure why this is needed)...
        lbe.push_front( BeamElement<HiddState>( ProbBack<HiddState>(), HiddState( Sign(hvBot,cFail,S_A), 0, EVar::eNil, K::kBot, jm.getResponse0(), StoreState(), Sign(hvBot,cFail,S_A) ) ) ); // no-fork, no-join?
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
