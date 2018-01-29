import re

class Dependency:
    def __init__(self, label='',head="", child=""):
        self.label = label
        self.head = head
        self.child = child

    def __str__(self):
        return self.label+"("+self.head+", "+self.child+")"
        
    def read(self, s):
        m = re.search("^([^\(]*)\(([^\,]*), ([^\)]*)\)", s)
        if m != None:
            self.label = m.group(1)
            self.head = m.group(2)
            self.child = m.group(3)
            
    def write(self):
            print(self.label+"("+self.head+", "+self.child+")")

    def is_same(self, d):
        if self.label == d.label and self.head == d.head and self.child == d.child:
            return True
        


#deps = ["Nmod(纪录-28, 世界-27)", "2(取得-10, 纪录-28)", "1(项-26, 纪录-28)", "2(打破-24, 纪录-28)","punc(打破-24, ；-29)"]
#
#for d in deps:
#    D = Dependency()
#    D.read(d)
#    print(D)
##    print(D.child)
#
#
#D = Dependency()
#D.head = "纪录-28"
#D.child = "世界-27"
#D.label= "Nmod"
#D.write()
#    
