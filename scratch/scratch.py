import numpy as np

a = "727969CE-C3DE-4204-B64A-350E7942AE98"
print(len(a))
b ="457DFB66-AC1D-402D-A7C7-DCC217832ACA"
print(len(b))

functions = []
for i in range(10):
    functions.append(lambda : i)

for f in functions:
    print(f())