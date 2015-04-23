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

////////////////////////////////////////////////////////////////////////////////

template<class M>
class ModelReader {
 private:
  M m;
 public:
  // Constructor methods...
  ModelReader ( int nArgs, char* argv[] ) : m() {

//    // Ignore flags...
//    for( int opt, option_index = 0; (opt = getopt_long(nArgs, argv, "b:v", long_options, &option_index)) != -1; );
    int a=1;
    for ( ; *argv[a]=='-'; a++ );

    // For each model file...
//    for ( int a=optind; a<nArgs; a++ ) {
    for ( ; a<nArgs; a++ ) {
      // Open file...
      //            FILE* pf = fopen(argv[a],"r"); assert(pf);
      ifstream fin ( argv[a], ios::in );
      if(!fin.good()){
        cerr << "Error loading model file " << argv[a] << endl;
        exit(1);
      }
      cerr << "Loading model \'" << argv[a] << "\'...\n";
      int line=1; char psBuff[1000];        //string sBuff;
      // For each line of model file...
      while ( fin.getline(psBuff,1000) ) {  //getline(cin,sBuff) ) {
        StringInput si(psBuff);           //si(sBuff.c_str());
        // Process comments or model lines...
        if ( !( psBuff[0]=='#' ||
                si>>m>>"\0"!=NULL ) )
          cerr<<"\nERROR: can't parse \'"<<psBuff<<"\' in line "<<line<<"\n\n";
        // Print progress (for big models)...
        if ( line%1000000==0 ) cerr<<"  "<<line<<" lines read...\n";
        line++;
      }
      fin.close();
      cerr << "Model \'" << argv[a] << "\' loaded.\n";
    }
  }

  // Extractor methods...
  const M& getModel ( ) const { return m; }
};


