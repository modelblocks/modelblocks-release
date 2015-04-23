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

#define SHORT

#include<stdio.h>
#include"nl-iomacros.h"
#include"nl-list.h"

static const char LEFT1 = '(' ;
static const char RIGHT1 = ')' ;
static const char LEFT2 = '<' ;
static const char RIGHT2 = '>' ;
//static const char LEFT2 = '{' ;
//static const char RIGHT2 = '}' ;
static const char LEFT3 = '<' ;
static const char RIGHT3 = '>' ;
static const char LEFT4 = '[' ;
static const char RIGHT4 = ']' ;

class Int
  {
  public:
    int i ;
  } ;

int main ( void )
  {
  int         c = getc(stdin) ;
  int         j = 0 ;
  List<Int>   liTabs ;

  //Listed(Int)* pi ;

  liTabs.push().i = 0 ;

  // Read in everything else...
  for ( ; c!=EOF && c!='\n' && c!=LEFT1 && c!=LEFT2 && c!=LEFT3 && c!=LEFT4 && /*c!=',' &&*/ c!=RIGHT1 && c!=RIGHT2 && c!=RIGHT3 && c!=RIGHT4; j++, c=getc(stdin) )
    putc ( c, stdout ) ;

  while ( c!=EOF )
    {
    // Newlines...
    if ( c=='\n' )
      {
      putc(c,stdout); j++; c=getc(stdin);
      j = 0 ;
      }
    // Open bracket...
    else if ( c==LEFT1 || c==LEFT2 || c==LEFT3 || c==LEFT4 )
      {
      putc(c,stdout); j++; c=getc(stdin);
      while ( c!=LEFT1 && c!=LEFT2 && c!=LEFT3 && c!=LEFT4 && c!=RIGHT1 && c!=RIGHT2 && c!=RIGHT3 && c!=RIGHT4 /*&& c!=','*/ && c!='\n' ) { putc(c,stdout); j++; c=getc(stdin); }
      liTabs.push().i = j ;
      }
    // Comma...
    else if ( c==',' )
      {
      #ifdef SHORT ////////////////////////////////
                                                 //
      putc(c,stdout); j++; c=getc(stdin);        //
                                                 //
      #else ///////////////////////////////////////

      putc(c,stdout); j++; c=getc(stdin);
      putc ( '\n', stdout ) ;

      j = 0 ;
      for ( ; j < liTabs.getFirst()->i; j++ )
        putc ( ' ', stdout ) ;

      for ( ; c==' '; c=getc(stdin) ) ;

      #endif //////////////////////////////////////
      }
    // Close bracket...
    else if ( c==RIGHT1 || c==RIGHT2 || c==RIGHT3 || c==RIGHT4 )
      {
      #ifdef SHORT ////////////////////////////////
      while ( c==RIGHT1 || c==RIGHT2 || c==RIGHT3 || c==RIGHT4 )
        {                                        //
        putc(c,stdout); j++; c=getc(stdin);      //
        liTabs.pop() ;                           //
        while ( c==' ' ) c=getc(stdin);          //
        }                                        //
      if ( c==',' )                              //
        {                                        //
        putc(c,stdout); j++; c=getc(stdin);      //
        }                                        //
      putc ( '\n', stdout ) ;                    //
                                                 //
      j = 0 ;                                    //
      for ( ; j < liTabs.getFirst()->i; j++ )    //
        putc ( ' ', stdout ) ;                   //
                                                 //
      for ( ; c==' '; c=getc(stdin) ) ;          //
      if ( c=='\n' ) c=getc(stdin) ;             //
                                                 //
      #else ///////////////////////////////////////

      putc(c,stdout); j++; c=getc(stdin);
      liTabs.pop() ;

      #endif //////////////////////////////////////
      }

    // Read in everything else...
    for ( ; c!='\n' && c!=LEFT1 && c!=LEFT2 && c!=LEFT3 && c!=LEFT4 && /*c!=',' &&*/ c!=RIGHT1 && c!=RIGHT2 && c!=RIGHT3 && c!=RIGHT4 && c!=EOF; j++, c=getc(stdin) )
      putc ( c, stdout ) ;
    }
  }
