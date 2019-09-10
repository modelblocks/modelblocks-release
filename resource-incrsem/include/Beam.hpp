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

#include <set>


template<class S>
class BeamElement;


template<class S>
class ProbBack : private pair<double, const BeamElement<S>*> {
 public :
  ProbBack ( );
  ProbBack ( double d , const BeamElement<S>& be );
  double                getProb ( ) const;
  const BeamElement<S>& getBack ( ) const;
  double&               setProb ( );
  void                  setBack ( const BeamElement<S>& );
  bool operator> ( const ProbBack<S>& ) const;
  bool operator< ( const ProbBack<S>& ) const;
};


template<class S>
class BeamElement : private pair<ProbBack<S>,S> {
 public:
  BeamElement ( )                                    : pair<ProbBack<S>,S> ( )        { }
  BeamElement ( const ProbBack<S>& pb, const S& hs ) : pair<ProbBack<S>,S> ( pb, hs ) { }
  double                getProb ( ) const;
  const BeamElement<S>& getBack ( ) const; 
  const S&              getHidd ( ) const { return pair<ProbBack<S>,S>::second; }
  const ProbBack<S>&    getProbBack ( ) const { return pair<ProbBack<S>,S>::first; }
  double&               setProb ( );
  void                  setBack ( const BeamElement<S>& );
  bool operator> ( const BeamElement<S>& ) const;
  bool operator< ( const BeamElement<S>& ) const;
  static const BeamElement<S> beStableDummy;
};


template<class S>
inline ProbBack<S>::ProbBack ( )                                     : pair<double, const BeamElement<S>*> ( 0.0, &BeamElement<S>::beStableDummy ) { }
template<class S>
inline ProbBack<S>::ProbBack ( double d , const BeamElement<S>& be ) : pair<double, const BeamElement<S>*> ( d,   &be                            ) { }
template<class S>
inline double                ProbBack<S>::getProb ( ) const { return  pair<double, const BeamElement<S>*>::first;  }
template<class S>
inline const BeamElement<S>& ProbBack<S>::getBack ( ) const { return *pair<double, const BeamElement<S>*>::second; }
template<class S>
inline double&               ProbBack<S>::setProb ( )       { return  pair<double, const BeamElement<S>*>::first;  }
template<class S>
void                        ProbBack<S>::setBack ( const BeamElement<S>& be ) { pair<double, const BeamElement<S>*>::second = &be; }
template<class S>
inline bool ProbBack<S>::operator> ( const ProbBack<S>& pb ) const { return pair<double, const BeamElement<S>*>( *this ) > pair<double, const BeamElement<S>*>( pb ); }
template<class S>
inline bool ProbBack<S>::operator< ( const ProbBack<S>& pb ) const { return pair<double, const BeamElement<S>*>( *this ) < pair<double, const BeamElement<S>*>( pb ); }


template<class S>
double                BeamElement<S>::getProb ( ) const { return pair<ProbBack<S>,S>::first.getProb(); }
template<class S>
const BeamElement<S>& BeamElement<S>::getBack ( ) const { return pair<ProbBack<S>,S>::first.getBack(); }
template<class S>
double&               BeamElement<S>::setProb ( )       { return pair<ProbBack<S>,S>::first.setProb(); }
template<class S>
void                  BeamElement<S>::setBack ( const BeamElement<S>& be ) { pair<ProbBack<S>,S>::first.setBack(be); }
template<class S>
inline bool BeamElement<S>::operator> ( const BeamElement<S>& be ) const { return pair<ProbBack<S>,S>( *this ) > pair<ProbBack<S>,S>( be ); }
template<class S>
inline bool BeamElement<S>::operator< ( const BeamElement<S>& be ) const { return pair<ProbBack<S>,S>( *this ) < pair<ProbBack<S>,S>( be ); }

template<class S>
const BeamElement<S> BeamElement<S>::beStableDummy = BeamElement<S> ( );


////////////////////////////////////////////////////////////////////////////////

template<class S>
class Beam : public set<BeamElement<S>,std::greater<BeamElement<S>>> {

 private:

  uint               iBeamWidth;
  map<S,ProbBack<S>> msp;

 public:

  Beam(uint i) : iBeamWidth(i) { }

  const pair<const S,ProbBack<S>>& get( const S& s ) const { 
    auto it = msp.find(s); 
    return ( it == msp.end() ) ? spDummy : *it ;
  }
  static const pair<const S,ProbBack<S>> spDummy;

  void tryAdd ( const S& s, ProbBack<S> p ) {
    // Only add if extra space or prob beats min...
    if( msp.size() < iBeamWidth || p.getProb() > set<BeamElement<S>,std::greater<BeamElement<S>>>::rbegin()->getProb() ) {
      const auto& isp = msp.find(s);
      // If state is already in beam, update prob...
      if( isp != msp.end() ) {
        if( p.getProb() > isp->second.getProb() ) {
          set<BeamElement<S>,std::greater<BeamElement<S>>>::erase( BeamElement<S>( isp->second, isp->first ) );
          set<BeamElement<S>,std::greater<BeamElement<S>>>::emplace( BeamElement<S>( p, s ) );
          msp.erase( isp );
          msp.emplace( s, p );
        }
      }
      // If state is new, update ss and prob...
      else {
        if( msp.size() >= iBeamWidth ) {
          const auto& ips = set<BeamElement<S>,std::greater<BeamElement<S>>>::rbegin();
          msp.erase( ips->getHidd() );
          set<BeamElement<S>,std::greater<BeamElement<S>>>::erase( *ips );
        }
        msp.emplace( s, p );
        set<BeamElement<S>,std::greater<BeamElement<S>>>::emplace( BeamElement<S>( p, s ) );
      }
    }
  }

  friend ostream& operator<< ( ostream& os, const Beam<S>& t ) {
    for( const auto& be : t )
      os << be.getProb() << " " << be.getHidd() << " me: " << &be << " myback: " << &be.getBack() << endl;
    return os;
  }
};

template<class S>
const pair<const S,ProbBack<S>> Beam<S>::spDummy = pair<const S,ProbBack<S>> ( S() , ProbBack<S>() ); 


