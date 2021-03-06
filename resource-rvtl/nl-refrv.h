////////////////////////////////////////////////////////////
  //template <class T>
template<class T, map<T,T>& domain>
class RefRV : public Id<const T*> {
 public:

  typedef RefRV<T,domain> BaseType;

  static const int NUM_VARS = 1;
  static const T   DUMMY;

  ////////////////////
  template<class P>
  class ArrayDistrib : public Array<pair<RefRV<T,domain>,P> > {
  };

  ////////////////////
  template<class P>
  class ArrayIterator : public pair<SafePtr<const ArrayDistrib<P> >,Id<int> > {
   public:
    static const int NUM_ITERS = NUM_VARS;
    operator RefRV<T,domain>() const { return ArrayIterator<P>::first.getRef().get(ArrayIterator<P>::second.toInt()).first; }
    //const DiscreteDomainRV<T,domain>& toRV() { return ArrayIterator<P>::first.getRef().get(ArrayIterator<P>::second.toInt()).first; }
  };

  // Constructor / destructor methods...
  RefRV ( )                 { Id<const T*>::set(NULL); }
  //RefRV ( int i )           { Id<const T*>::set(i); }
  //RefRV ( const T& t )      { Id<const T*>::set(&t);   }
  RefRV ( const T& t )      { if(domain.find(t)==domain.end()) *(const_cast<T*>(Id<const T*>::set(&domain[t]).toInt())) = t;
                              else Id<const T*>::set(&domain[t]); }

  // Specification methods...
  template<class P>
  RefRV<T,domain>& setVal ( const ArrayIterator<P>& it ) { *this=it; return *this; }
  //T&       setRef ( )       { return Id<const T*>::setRef(); }

  // Extraction methods...
  const T& getRef ( ) const { return (Id<const T*>::toInt()==NULL) ? DUMMY : *(static_cast<const T*>(Id<const T*>::toInt())); }
  static map<T,T>& setDomain ( ) { return domain; }

  // Input / output methods..
  friend ostream& operator<< ( ostream& os, const RefRV<T,domain>& rv ) { return os <<&rv.getRef(); }  //{ return  os<<rv.getRef(); }
  friend String&  operator<< ( String& str, const RefRV<T,domain>& rv ) { return str<<"addr"<<(long int)(void*)&rv.getRef(); }  //{ return str<<rv.getRef(); }
  friend pair<StringInput,RefRV<T,domain>*> operator>> ( const StringInput ps, RefRV<T,domain>& rv ) { return pair<StringInput,RefRV<T,domain>*>(ps,&rv); }
  friend StringInput operator>> ( pair<StringInput,RefRV<T,domain>*> delimbuff, const char* psDlm ) {
    if (StringInput(NULL)==delimbuff.first) return delimbuff.first;
    return NULL; //psIn;
  }
};
template <class T, map<T,T>& domain> const T RefRV<T,domain>::DUMMY;
