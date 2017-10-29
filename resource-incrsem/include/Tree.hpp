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

char psOpenParen[]  = "(";
char psCloseParen[] = ")";

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<string> L;

////////////////////////////////////////////////////////////////////////////////

template<class D>
class Tree : public D, public DelimitedList<psX,Tree<D>,psSpace,psX> {
// private:
//  D d;
 public:
  Tree ( )                : D( ),    DelimitedList<psX,Tree<D>,psSpace,psX>( ) { }
  Tree ( const char* ps ) : D( ps ), DelimitedList<psX,Tree<D>,psSpace,psX>( ) { }    // NOTE: This only creates leaf nodes.
  friend pair<istream&,Tree<D>&> operator>> ( istream& is, Tree& t ) {
    return pair<istream&,Tree<D>&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,Tree<D>&> ist, const char* psDelim ) {
    if ( ist.first.peek()=='(' ) return ist.first >> "(" >> (D&)ist.second >> " " >> (DelimitedList<psX,Tree<D>,psSpace,psX>&)ist.second >> ")" >> psDelim;
    else                         return ist.first >> (D&)ist.second >> psDelim;
  }
  friend bool operator>> ( pair<istream&,Tree<D>&> ist, const vector<const char*>& vpsDelim ) {
    if ( ist.first.peek()=='(' ) return ist.first >> "(" >> (D&)ist.second >> " " >> (DelimitedList<psX,Tree<D>,psSpace,psX>&)ist.second >> ")" >> vpsDelim;
    else                         return ist.first >> (D&)ist.second >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const Tree<D>& t ) {
    if ( t.size()>0 ) return os << psOpenParen << (D&)t << psSpace << (DelimitedList<psX,Tree<D>,psSpace,psX>&)t << psCloseParen;
    else              return os << (D&)t;
  }
//  operator const D ( ) const { return (D&)t; }
};


