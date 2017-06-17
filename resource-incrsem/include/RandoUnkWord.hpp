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

#ifndef W__
#define W__
DiscreteDomain<int> domW;
class W : public Delimited<DiscreteDomainRV<int,domW>> {
 public:
  W ( )                : Delimited<DiscreteDomainRV<int,domW>> ( )    { }
  W ( int i )          : Delimited<DiscreteDomainRV<int,domW>> ( i )  { }
  W ( const char* ps ) : Delimited<DiscreteDomainRV<int,domW>> ( ps ) { }
};
typedef W ObsWord;
#endif

////////////////////////////////////////////////////////////////////////////////

W unkWord ( const char* ps ) {
  return ( 0==strcmp(ps+strlen(ps)-strlen("ing"), "ing") ) ? W("!unk!ing") :
         ( 0==strcmp(ps+strlen(ps)-strlen("ed"),  "ed" ) ) ? W("!unk!ed") :
         ( 0==strcmp(ps+strlen(ps)-strlen("s"),   "s"  ) ) ? W("!unk!s") :
         ( 0==strcmp(ps+strlen(ps)-strlen("ion"), "ion") ) ? W("!unk!ion") :
         ( 0==strcmp(ps+strlen(ps)-strlen("er"),  "er" ) ) ? W("!unk!er") :
         ( 0==strcmp(ps+strlen(ps)-strlen("est"), "est") ) ? W("!unk!est") :
         ( 0==strcmp(ps+strlen(ps)-strlen("ly"),  "ly" ) ) ? W("!unk!ly") : 
         ( 0==strcmp(ps+strlen(ps)-strlen("ity"), "ity") ) ? W("!unk!ity") : 
         ( 0==strcmp(ps+strlen(ps)-strlen("y"),   "y"  ) ) ? W("!unk!y") : 
         ( 0==strcmp(ps+strlen(ps)-strlen("al"),  "al" ) ) ? W("!unk!al") :
         ( ps[0]>='A' && ps[0]<='Z'                      ) ? W("!unk!cap") :
         ( ps[0]>='0' && ps[0]<='9'                      ) ? W("!unk!num") :
                                                             W("!unk!");
}

