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

class Tree : public DelimitedList<psX,Tree,psSpace,psX> {
 private:
  L l;
 public:
  Tree ( )                : DelimitedList<psX,Tree,psSpace,psX> ( )        { }
  Tree ( const char* ps ) : DelimitedList<psX,Tree,psSpace,psX> ( ), l(ps) { }
  friend pair<istream&,Tree&> operator>> ( istream& is, Tree& t ) {
    return pair<istream&,Tree&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,Tree&> ist, const char* psDelim ) {
    if ( ist.first.peek()=='(' ) return ist.first >> "(" >> ist.second.l >> " " >> (DelimitedList<psX,Tree,psSpace,psX>&)ist.second >> ")" >> psDelim;
    else                         return ist.first >> ist.second.l >> psDelim;
  }
  friend bool operator>> ( pair<istream&,Tree&> ist, const vector<const char*>& vpsDelim ) {
    if ( ist.first.peek()=='(' ) return ist.first >> "(" >> ist.second.l >> " " >> (DelimitedList<psX,Tree,psSpace,psX>&)ist.second >> ")" >> vpsDelim;
    else                         return ist.first >> ist.second.l >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const Tree& t ) {
    if ( t.size()>0 ) return os << psOpenParen << t.l << psSpace << (DelimitedList<psX,Tree,psSpace,psX>&)t << psCloseParen;
    else              return os << t.l;
  }
  operator const L ( ) const { return l; }
};


