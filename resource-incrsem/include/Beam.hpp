
#include <set>

template<class P,class S>
class Beam : public set<pair<P,S>,std::greater<pair<P,S>>> {

 private:

  uint     iBeamWidth;
  map<S,P> msp;

 public:

  Beam(uint i) : iBeamWidth(i) { }

  const pair<const S,P>& get( const S& s ) { return *msp.find(s); }

  void tryAdd ( const S& s, P p ) {
    // Only add if extra space or prob beats min...
    if( msp.size() < iBeamWidth || p.first > set<pair<P,S>,std::greater<pair<P,S>>>::rbegin()->first.first ) {
      const auto& isp = msp.find(s);
      // If state is already in beam, update prob...
      if( isp != msp.end() ) {
        if( p > isp->second ) {
          set<pair<P,S>,std::greater<pair<P,S>>>::erase( pair<P,S>( isp->second, isp->first ) );
          set<pair<P,S>,std::greater<pair<P,S>>>::emplace( pair<P,S>( p, s ) );
          msp.erase( isp );
          msp.emplace( s, p );
        }
      }
      // If state is new, update ss and prob...
      else {
        if( msp.size() >= iBeamWidth ) {
          const auto& ips = set<pair<P,S>,std::greater<pair<P,S>>>::rbegin();
          msp.erase( ips->second );
          set<pair<P,S>,std::greater<pair<P,S>>>::erase( *ips );
        }
        msp.emplace( s, p );
        set<pair<P,S>,std::greater<pair<P,S>>>::emplace( pair<P,S>( p, s ) );
      }
    }
  }

  friend ostream& operator<< ( ostream& os, const Beam<P,S>& t ) {
    for( const auto& be : t )
      os << be.first.first << " " << be.second << endl;
    return os;
  }
};


