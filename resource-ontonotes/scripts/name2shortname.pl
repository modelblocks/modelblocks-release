# substitutes 'ENAMEX TYPE' with 'ENAMEXTYPE' for tokenization
s/ENAMEX TYPE/ENAMEXTYPE/g;

# removes S_OFF="." and E_OFF="."
s/ S_OFF="[1-9]+">/>/g;
s/ E_OFF="[1-9]+">/>/g;
