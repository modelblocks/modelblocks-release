#include <iomanip>
#include <sstream>
#include <string>

std::string ZeroPadNumber(int width, int num){
        std::ostringstream ss;
        ss << std::setw( width ) << std::setfill( '0' ) << num;
        return ss.str(); 
}

