def a(x):
    return "a(%s)" % (x)

def b(f,x):
    return f(x)

print(b(a,10))