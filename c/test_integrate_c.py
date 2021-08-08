import os
# import of wrapper functions
import c_wrapper 

# example run of c integrated code
test_list = [34, 2, 1, 1]


old = [3, 7, 5, 10]
new = [4, 5, 7, 7]
real = 10

print(c_wrapper.disagreement(test_list))
print(c_wrapper.consensus(old, new, real))
