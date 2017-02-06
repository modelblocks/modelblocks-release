///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. //
//                                                                           //
//    ModelBlocks is free software: you can redistribute it and/or modify    //
//    it under the terms of the GNU General Public License as published by   //
//    the Free Software Foundation, either version 3 of the License, or      //
//    (at your option) any later version.                                    //
//                                                                           //
//    ModelBlocks is distributed in the hope that it will be useful,         //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//    GNU General Public License for more details.                           //
//                                                                           //
//    You should have received a copy of the GNU General Public License      //
//    along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

template<class WY>
class BestFirstBeam : public Array<WY> {
 private:
  SimpleHash<typename WY::VAR_TYPE,bool> h;
 public:
  BestFirstBeam ( )       : Array<WY>(),  h() { }
  BestFirstBeam ( int i ) : Array<WY>(i), h() { }
  bool add ( const WY& wy ) {
    if ( !h.contains(wy.getY()) ) {
      Array<WY>::add() = wy;
      h.set(wy.getY()) = true;
      return true;
    }
    return false;
  }
};

template<class WY>
class BackPointerBestFirstBeam : public BestFirstBeam<WY> {
 private:
  Array<int> ab;
 public:
  BackPointerBestFirstBeam ( )       : BestFirstBeam<WY>(),  ab()  { }
  BackPointerBestFirstBeam ( int i ) : BestFirstBeam<WY>(i), ab(i) { }
  void add ( const WY& wy, int iB ) {
    if ( BestFirstBeam<WY>::add(wy) ) ab.add()=iB;
  }
  int getBack ( int i ) const { return ab[i]; }
};

