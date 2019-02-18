
# coding: utf-8

# In[ ]:

# http://people.duke.edu/~ccc14/pcfb/numpympl/NumpyBasics.html
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
# http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf
# http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html
# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

# NUMPY TO MATLAB - https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html


# In[ ]:

import numpy as np


# ## NDARRAY
# An ndaray is a n-dimensional array where all items are of the same type (unlike a Python data structure) and consequently use the same amount of space. There are 21 different types of objects (also called dtypes) that can be stored in ndarray. They are
# * bool_ 
# * byte
# * short
# * intc
# * int_
# * longlong
# * intp
# * ubyte
# * ushort
# * uintc
# * uint
# * ulonglong
# * uintp
# * single
# * float_
# * longfloat
# * csingl
# * complex_
# * clongfloat
# * object_
# * str_
# * unicode_
# * void
# 
# For some of the dtypes, a _ to differentiate that dtype from the corresponding Python type. Such types are also called as 'enhanced scalars).  They have the same precision as the Python type.
# 
# All the types except the str_, unicode_ and void are of fixed size. 

# In[ ]:

# Creating a simple ndarray
a = np.arange(8) # similar to range(8)
print(a, type(a))


# In[ ]:

# Indexing
print(a[3], type(a[3])) # since there is only one value, its type is the type of each element
print(a[2:5], type(a[2:5])) # The slicing results in an ndarray


# In[ ]:

# Universal functions or ufunc
# They perform element by element operation on an ndarray.
b = np.sin(a)
print(b)

c = np.add(a, b)
print(c)

# For a full list of ufunc, visit 
# http://docs.scipy.org/doc/numpy/reference/ufuncs.html


# In[ ]:

# In the case of add function, a and b both had same sized vector.
# What happens if they are of different sizes as in the example below
d = np.add(a, 3)
print(d)

# The meaning of adding a which is a vector to a scalar 3 is done by 
# adding each element in a with the value 3. In otherwords, the value 3
# was 'broadcast' to each element in a and added.

# NEED MORE EXAMPLES


# In[ ]:

# NDARRAY attributes

print(b.shape) # Size of the matrix
print(b.ndim) # Number of dimensions
print(b.dtype) # Data type of each element
print(b.itemsize) # Memory occupied by each element
print(type(b))  # Type of b
print(dir(b.dtype))


# In[ ]:

print(b.flags) 


# In[ ]:

# Check if data is little or big endian
# < for little endian and > for big endian
print(b.dtype.str)


# ### ARRAY CONVERSION

# In[ ]:

print(b.tolist()) # convert ndarray b to list


# In[ ]:

# Write the vector b to a csv file with 3 precision
b.tofile(file="data.csv", sep=",", format="%0.3f")


# In[ ]:

b.dump('data.pickle') # Write to pickle file


# ### ARRAY SHAPE 

# In[ ]:

a = np.arange(8)
b = a.reshape((2, 4), order='C') # Change the matrix to size 2x4
print(b)
# Modify the order from 'C' to 'F'


# In[ ]:

c = a.resize((2, 4))
print(c) # None as resize modifies inplace while reshape creates a new array
print(a)


# In[ ]:

a = np.arange(8).reshape((2, 4), order='C' )
flat_a = a.flatten()
print(flat_a) # Convert to 1D vector
flat_a[0]= 6
print(flat_a)
print(a) 
# Since flat_a is a new copy, any change to flat_a does not affect a


# In[ ]:

a = np.arange(8).reshape((2, 4), order='C' )
ravel_a = a.ravel() # Convert to 1D vector
print(ravel_a)
ravel_a[0]= 6
print(ravel_a)
print(a)
# Since ravel_a is NOT a copy, any change to ravel_a affects a


# In[ ]:

# Shallow copy
d = a
a[0][0] = 4
print(a)
print(d) # Note d and a will have same values even though we modified only a


# In[ ]:

# Deep copy
d = a.copy()
a[0][0] = 3
print(a)
print(d) # Note d and a will have same values even though we modified only a


# In[ ]:

# INCLASS ACTIVITY
'''
A power ball needs a list of 6 numbers. The first 5 numbers have value between 1 and 59.
The last number also called power ball number will be between 1 and 35. 
Write a Python program to create this list with 6 numbers. Modify the code
so that it is seeded by the current date. 
'''


powerball = np.floor((np.random.rand(1, 6)*59)+1.0)
powerball[0, 5] = np.floor((np.random.rand(1, 6)*35)+1.0)[0, 0]
print(powerball)


# ## ARRAY MANIPULATION

# In[ ]:

import numpy as np
a = np.random.rand(2, 4)
print(a)
a.sort() # sort(axis=-1, kind=’quick’, order=None)
print(a)


# In[ ]:

a = np.random.rand(2, 4)
print(a)
print(a.argsort()) # argsort(axis=-1, kind=’quick’, order=None)


# In[ ]:

a = np.random.rand(2, 4)*2
a = a.astype('int')
print(a)
print(a.nonzero())


# ### ARRAY CALCULATIONS

# In[ ]:

import numpy as np
a = np.random.rand(2, 2)*5
b = a.astype('int')
print(b)
print('Any element is {0}'.format(b.any()))
print('Sum of all elements is {0}'.format(b.sum()))
print('The product of all element is {0}'.format(b.prod()))
print('The max of all element is {0}'.format(b.max()))
print('The product of all element is {0}'.format(b.prod()))


# ### ARRAY INDEXING

# In[ ]:

# Basic slicing
import numpy as np
a = np.random.rand(10, 10)*5
b = a.astype('int')
print(b)
print('The rows=1 and cols=2 element is {0}'.format(b[1,2])) 
print('The first row is {0}'.format(b[:,0])) # rows, cols. all rows for cols=0
print('The third row is {0}'.format(b[2, :])) # all cols for rows = 2


# ### ROUTINES

# In[ ]:

# array(object=, dtype=None, copy=True, order=None, subok=False, ndmin=0)
# Convert any object to a ndarray. If copy is set to True, then a new copy is made.
# Convert a Python list or tuple to numpy array
import numpy as np

c = np.array((4, 5, 6), dtype=np.float32) # Change this to int and see the output
print(c, type(c), c.dtype)

# There is another method called 'asarray' which is same as 'array' except
# that the copy defaults to False.


# In[ ]:

# Will create a linear list of vaues starting from 'start' and ends at 'stop-1'
# in steps of 'step'
d = np.arange(start=10, stop=20, step=2, dtype=np.float32)
print(d)


# In[ ]:

# Create an empty array (i.e., uninitalized array)
f = np.empty(shape=(2, 2), dtype=np.int32, order='C')
print(f)


# In[ ]:

d = np.zeros(shape=(3, 4), dtype=np.int64)
print(d)
print(d.itemsize, d.dtype)


# In[ ]:

e = np.ones(shape=(3, 3), dtype=np.int32)
print(e)


# In[ ]:

f = np.identity(n=3, dtype=np.int32)
print(f)


# In[ ]:

g = np.random.rand(3, 3)*5
print(g)
print(np.where(g)) # Returns the x, y coordinates


# In[ ]:

k = np.random.rand(3, 3)*5
# Need to squeeze to convert it 1D array for correlation
print('The correlation is {0}'.format(np.correlate(g.flatten(), k.flatten(), mode='full')))
print('The convolution is {0}'.format(np.convolve(k.flatten(), g.flatten())))
print('The dot product is {0}'.format(np.dot(k, g)))
print('The cross product is {0}'.format(np.cross(k, g)))


# In[ ]:

m = np.random.rand(100, 100)*100
m = m.astype('int')
print('The histogram is {0}'.format(np.histogram(m)))
# The first array is the frequency and the second array is the bin


# In[ ]:

# Create 10 numbers between 1 and 20
p = np.linspace(start = 1, stop = 20, num=10)
print('The linear space of value is {0}'.format(p))

p = np.logspace(start = 0.1, stop = 0.20, num=10)
print('The log space of value is {0}'.format(p))


# ### OPERATIONS

# In[ ]:

import numpy as np

c = np.array((4, 5, 6), dtype=np.float32)
d = np.linspace(start=10, stop=13, num=3)
print(c)
print(d)
f = d-c # Subtract matrix
print(f, f.dtype)


# In[ ]:

f = 10*c # Multiply a matrix with scalar. The matrix e is of dtype=int64 but the final matrix is of dtype=float64
print f, f.dtype


# In[ ]:

h = f > 50 # Compare every element with the value of 0.5
print h


# In[ ]:

# Indexing with boolean arrays
k = f > 50
print(k)
print f[k] # Returns values in p that are True in t
print f[f>50] # This is same as the previous line except that we are not creating a new array t


# #### Calculate value of pi using Gregory-Leibniz series
# $$ 1\,-\,{\frac {1}{3}}\,+\,{\frac {1}{5}}\,-\,{\frac {1}{7}}\,+\,{\frac {1}{9}}\,-\,\cdots \;=\;{\frac {\pi }{4}}.\! $$

# In[ ]:

noofterms = 100000

# Create numerator array
numerator = np.ones(shape=(1, noofterms))
# Change alternate values from +1 to -1
numerator[0, 1::2] = -1
# Denominator = 1, 3, 5, 7 ...

# Sum all terms and multiply by 4
denominator = np.linspace(1, noofterms*2-1, noofterms)
pival = 4.0*np.sum(numerator/denominator)
print('The value of pi is {0}'.format(pival))


# In[ ]:



