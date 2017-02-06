
# 'all_of' quantifier
def all_of(L,f):
    for l in L:
        if not f(l): return False
    return True

# 'none_of' quantifier
def none_of(L,f):
    for l in L:
        if f(l): return False
    return True

# 'some_of' quantifier
def some_of(L,f):
    for l in L:
        if f(l): return True
    return False

