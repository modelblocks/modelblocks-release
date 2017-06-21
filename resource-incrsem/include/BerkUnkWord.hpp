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

W unkWordBerk (const char* ps ) {
    // the following code ripped from berkeley sophisticated lexicon
    string s = string("UNK");
    string word = std::string(ps); //.getString();
    size_t wlen = word.length();
    int numCaps = 0;
    bool hasDigit = false;
    bool hasDash = false;
    bool hasLower = false;
    for(size_t i = 0; i < wlen; i++) {
        char ch = word[i];
        if(isdigit(ch)) {
            hasDigit = true;
        } else if(ch == '-') {
            hasDash = true;
        } else if(isalpha(ch)) {
            if(islower(ch)) {
                hasLower = true;
            } else {
                numCaps++;
            }
        }
    }
    char ch = word[0];
    string lowered = std::string(ps); //.getString();
    for(size_t i = 0; i < wlen; i++) {
        lowered[i] = tolower(lowered[i]);
    }
    if(isupper(ch)) {
        if(numCaps == 1) {
            s += "-INITC";
            //X low = X(lowered.c_str());
            //	if(this->get(f).contains(low)) {
            //  s += "-KNOWNLC";
            //}
        } else {
            s += "-CAPS";
        }
    } else if(!isalpha(ch) && numCaps > 0) {
        s += "-CAPS";
    } else if(hasLower) {
        s += "-LC";
    }
    if(hasDigit) {
        s += "-NUM";
    }
    if(hasDash) {
        s += "-DASH";
    }
    if(lowered.rfind("s") == (wlen-1) && wlen >= 3) {
        char ch2 = lowered[wlen-2];
        if(ch2 != 's' && ch2 != 'i' && ch2 != 'u') {
            s += "-s";
        }
    } else if(wlen >= 5 && !hasDash && !(hasDigit && numCaps > 0)) {
        if(lowered.rfind("ed") == (wlen-2)) {
            s += "-ed";
        } else if(lowered.rfind("ing") == (wlen-3)) {
            s += "-ing";
        } else if(lowered.rfind("ion") == (wlen-3)) {
            s += "-ion";
        } else if(lowered.rfind("er") == (wlen-2)) {
            s += "-er";
        } else if(lowered.rfind("est") == (wlen-3)) {
            s += "-est";
        } else if(lowered.rfind("ly") == (wlen-2)) {
            s += "-ly";
        } else if(lowered.rfind("ity") == (wlen-3)) {
            s += "-ity";
        } else if(lowered.rfind("y") == (wlen-1)) {
            s += "-y";
        } else if(lowered.rfind("al") == (wlen-2)) {
            s += "-al";
        }
    }
    // cerr << "Converting " << word << " to " << s << endl;
    return W(s.c_str()); //.c_str();
}
//  return ( 0==strcmp(ps+strlen(ps)-strlen("ing"), "ing") ) ? W("!unk!ing") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("ed"),  "ed" ) ) ? W("!unk!ed") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("s"),   "s"  ) ) ? W("!unk!s") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("ion"), "ion") ) ? W("!unk!ion") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("er"),  "er" ) ) ? W("!unk!er") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("est"), "est") ) ? W("!unk!est") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("ly"),  "ly" ) ) ? W("!unk!ly") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("ity"), "ity") ) ? W("!unk!ity") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("y"),   "y"  ) ) ? W("!unk!y") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("al"),  "al" ) ) ? W("!unk!al") :
//         ( ps[0]>='A' && ps[0]<='Z'                      ) ? W("!unk!cap") :
//         ( ps[0]>='0' && ps[0]<='9'                      ) ? W("!unk!num") :
//                                                             W("!unk!");
//}
