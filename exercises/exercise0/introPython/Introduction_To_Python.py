
# coding: utf-8

# #                                    Introduction to Python, Numpy and Scipy

# ### Tewodros Amberbir, Christiano Gava

# During the previous offerings of the course, the exercises have been mainly handed in MATLAB. Now we have decided to change the language of the exercise submission to python. Python, apart from being convenient for quick prototyping, it has numerous open-source libraries supporting scientific computing. 
# 
# Since some students come to this lecture before learning Python, we have decided to do a quick introduction to the Python Language and some of the libraries we will use in the upcoming exercises.

# ## Tasks for this Exercise
# 
# In this exercise you are expected to accomplish the following goals:
# 
# 1. Installing Python, pip, scipy, numpy, matplotlib packages.
# 2. Learning basics of the Python programming language and some useful packages.

# # Installation Instructions
# 
# Installation instructions for Linux users. <br>
# 
# ### Python
# **sudo apt-get update** <br>
# **sudo apt-get install python2.7** <br>
# ### Pip
# ** wget https://bootstrap.pypa.io/get-pip.py ** <br>
# ** sudo python get-pip.py** <br>
# ### Scipy, Numpy and Matplotlib
# ** sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev ** <br>
# ** sudo pip install scipy ** <br>
# ** sudo pip install numpy** <br>
# ** sudo pip install matplotlib ** <br>

# # Python

# Python is an interpreted high-level programming language. It was developed by Guido van Rossum as successor to the ABC language. ABC was designed to be a language for prototyping and teaching. Python shares many features with ABC, like *not having variable declaration *, the **off-side-rule**(expressing blocks via indentation), etc.
# 
# Python was developed with the following main goals:
#     - easy and intuitive to learn; at the same time to be as powerful as other languages.
#     - readable code
#     - short development time and 
#     - open source
# 

# In[18]:


# Zen of Python by Tim Peters
# Uncomment the following line
# import this


# ## Built-in Data types
# Python supports the following primitive data types: 
#     - Integers, int,
#     - Floating-point, float
#     - Complex, complex
#     - String, str

# ## Variables

# In[2]:


# Lets create a string variable
str_msg = "Hello World!!"
print(str_msg)
print(type(str_msg))


# In[3]:


# Integers and Floats
a = 1
print(a)
print(type(a))


# In[1]:


# Note that there is no need for Variable declaration
# In Python, data type is inferred during assignment, see how a float variable is initialized

b = 1.0
print(b, type(b))


# In[2]:


# Creating a complex Variable, complex(real, imaginery)

c = complex(10,20)
print(c)


# ## Builtin Collection Types

# ### List
# List is an ordered set of Python objects. Imagine we have 3 cameras, camera 1, camera 2 and camera 3. It is might be convenient to store their focal length values in a list: focal_lengths = [f1, f2, f3].

# In[5]:


# Lists are created via comma sparated objects inside square brackates

# An empty list
list_0 = []
list_1 = [1,2,3]
print(list_0)
print(list_1)

# Elements of a list could belong to different classes, including list itself
# A List of integer, float, string, list
mixed_list = [1, 2.0, 3, "four" , []]
print(mixed_list)


# **list.pop(i)** <br>
# Pops the element at index i. If no index argument is passed it pops the last element.

# In[6]:


list_1 = [1,2,3,4]
print("List_1 == ", list_1)
a = list_1.pop(0)
print("List_1 == ", list_1)
print(a)
b = list_1.pop()
print(b)


# **list.index(x)** <br>
# Returns the **index** in the list of the first item whose value is **x**. It is an error if there is no such
# item.

# In[7]:


list_1 = [1,2,3,4]
idx = list_1.index(3)
print("Index == ", idx)


# ### Tuples
# Tuples are **immutable** sequences. They are similar to lists except that
# it is not possible to change them once they are created. Whenever you have a sequence of objects that will not change throughout execution of your program, it is usually a good idea to use tuples instead of lists.

# In[8]:


# An empty tuple is created with two parentheses
a = ()

a = (0,1,2,3,"four")
print(a[0])
print(a[4])

print("Casting Lists to Tuples")
list_1 = [0,1,2,3,"four"]
tuple_1 = tuple(list_1)
print(list_1)
print(tuple_1)
# Trying to change element of a tuple will give you an error
a[4] = 4


# ### Dictionaries
# Dictionaries are unordered collection of ***key:value*** pairs. Dictionaries are similar to what is called **associative arrays**. Unlike lists, where indexing is possible with integer indexes, dictionaries allow us to use strings or other objects as keys.

# In[3]:


dict_empty = {}
dict_1 = {"one": 1, "two":2, "x": 3}
print(dict_1)
print(dict_1['two'])


# In[19]:


# Keys in a dictionary must be unique

dict_1 = {"one": 1, "one": 3}
print(dict_1)
print(dict_1['one'])

dict_2 = {"one": 10, "two": 20, "three":30}
print(dict_2)
print("Let's remove one")
dict_2.pop("one")
print(dict_2)
print("Let's remove three also")
dict_2.pop("three")
print(dict_2)


# ### Sets
# **Set** is another Python collection type. Set could be understood as unordered collection of unique
# objects. Sets have methods that can perform set operations like **union()**, **intersection()**, **issubsset()**.

# ## File IO

# ### Text Files

# Python offers a bult-in ***file*** object that lets us access files.

# In[20]:


# Let's write a string to a file
fid = open('file_1.txt', 'w')
fid.write("Hello World!")
fid.close()


# In[22]:


# Open the file and print its content
fid = open('file_1.txt', 'r')
text = fid.read()
fid.close()
print(text)


# ##### Read on the internet how to read, CSV and JSON files. SInce these are common file formats in practice.

# # Numpy

# Numpy is a Python library that offers efficient methods for numerical operations on muti-dimensional arrays. Numpy's array class is called ndarray or array.

# In[24]:


# Importing numpy
import numpy as np
arr = np.array(()) # Empty array
print("Empty array ", arr)


# In[25]:


# Let's create a numpy array that contains something
arr_1 = np.array([[0,1,2,3],[1,2,3,4],[3,4,5,6]])
print(arr_1)

# Numpy arrays have shape attribute
print(arr_1.shape)


# In[26]:


# Numpy has functions that can let us create special types of matrices
ones = np.ones([2,2]) # the argument [2,2] is the size of the 
zeros = np.zeros((3,3))
rand = np.random.rand(2,2)
print("\n ones of size 2x2, \n ", ones)

print("\n 2x2 matrix of random numbers, \n ", rand)


# In[27]:


full = np.full((3,3),5) # A 3x3 Matrix of 5s
print("\n", full)
eye  = np.eye(3) # 3x3 Identity matrix
print("\n", eye)


# ### Array Indexing

# Numpy has a number of easy ways to access elements of an array or slice arrays. 

# In[28]:


arr_1 = np.array([[0,1,2,3],[1,2,3,4],[3,4,5,6]])


# accessing elements of a matrix

print("row=0, col=1 \n", arr_1[0,1]) #prints 1
print("row=0, col=2 \n", arr_1[0,2]) #prints 2

print("row=2, col=3 \n", arr_1[2,3]) #prints 6

# Let's access the first row

row_0 = arr_1[0,:]
print(arr_1)
print("first_row ", row_0)

# : means "all", or start:end
row_0 = arr_1[0,0:3]
print("\n again first_row \n ", row_0)
# There are few more varations to this way of accessing arrays

# :3,means start:3
row_0 = arr_1[0,:3]
print("\n again first_row \n ", row_0)

# 0:, means 0:end
row_0 = arr_1[0,0:]
print("\n again first_row \n ", row_0)

# Let's print the first column
col_0 = arr_1[:,0]
print("first_col ", col_0)

# You can also access sub-array as follows
# Let's retreive a subarray that spans: 
# rows: 0 and 1 & columns: 1,2 and 3

sub_matrix = arr_1[0:2,1:4]
print("\n Rows 0,1 and Cols 1,2,3 \n", sub_matrix)



# ### Boolean Array Indexing

# Boolean array indexing enables accessing an array via another array of boolean values. Boolean indexing is a very useful feature of Numpy arrays. At times we are interested to index arrays based on a **condition**. For instance, if we have an arra, which contains depth values for a 3x3 pixel region we might want to slice out the entries with negative depth values, or discard entries with very large depth values, etc.

# In[29]:


# Boolean indexing
arr = np.array([1,2,3,4])
flag = np.array([False, False, False, True])
print(arr[flag]) #Prints [4]

D = np.array([[1,-3,4],[-5,6,4],[9,9,-9]])
print("\n", D, "\n")

# Let's replace all negative values to 0


flag = D<0 # This creates an a boolean array of same shape as D,
           # assigns True if the condition is satisfied, False otherwise

D[flag] = 0
print("Flag  \n", flag, "\n")
print("\n", D)


# #### Numpy Array Maths
# Numpy provides a number of mathematical operations that we can perform on matrices.
# 
# ***array.min()*** returns mininimum value of the matrix, <br>
# ***array.min(axis=dim)*** returns minimum value across the axis give as an argument <br>
# 
# ***array.max(axis=dim)***  max <br>
# ***array.mean(axis=dim)*** Mean <br>
# ***array.std(axis=dim)***  Standard deviation <br>
# 
# ##### Elementwise operations
# ***np.log(array)*** Natural Logarithm <br>
# ***np.log10(array)***                                         Logarithm base 10 <br>
# ***np.exp(array)***                                         Exponential of all elements oa the array <br>
# 
# Numpy has a large number of methods. You can just search the collection of methods using ***np.lookfor(keyword)*** method.

# In[20]:


a = np.random.rand(3,3)
print("\n", a)
print("\n", a.sum(axis=0)) # sum, across the given axis
print("\n", a.sum()) # sum of all array elements

print("\n", a.min(axis=0))
print("\n", a.min())


# #### Manipulating Arrays
# 
# Numpy offers powerful array manipulation options via methods such as ***transpose()***, ***append()***, ***reshape()*** etc. <br>

# In[21]:


# Tanspose
array_0 = np.random.rand(3,2)
print("array_0.shape")
print(array_0.shape)

array_t = array_0.transpose()
print("array_t.shape")
print(array_t.shape)

# Reshaping
print("array_0.reshape(6,1)")
print(array_0.reshape(6,1))


# In[30]:


# Transpose operation allows changing oder-of-dimensions
# Suppose we have an array of [H, W, D], if we want to re-order the dimensions so that D is 
# the first dimension of the array, i.e to have the array dimensions as [D, H, W]

array_0 = np.random.rand(2,3,4)
print("array_0.shape")
print(array_0.shape) # [2,3,4]
array_1 = array_0.transpose(2,0,1)
print("array_0.transpose(2,0,1).shape")
print(array_1.shape) # [4,2,3]

# Sometimes we would like to squeeze an ndarray on to a one dimensional vector 
print("2D Array")
print(array_0.shape)
print("1D Vector")
array_0 = np.array([[1,2],[3,4],[5,6]])
array_1d = array_0.ravel()
print(array_1d.shape)


# #### Stacking Arrays

# In[25]:


array_0 = np.random.rand(2,2)
array_1 = np.random.rand(2,2)

# Let's stack array_0 and array_1 in to a [2,2,2] array
array_0_1 = np.stack((array_0, array_1))
print(array_0_1.shape)


# In[26]:


# Horizontal and vertical stacking

array_0 = np.random.rand(2,2)
array_1 = np.random.rand(2,2)

h_stack = np.hstack((array_0, array_1))
v_stack = np.vstack((array_0, array_1))
print("arrays, ", array_0.shape, array_1.shape)
print("horizontal stack, ", h_stack.shape)
print("vertical stack, ",v_stack.shape)


# In[27]:


# Stacking across a specific dimension

array_0 = np.random.rand(1,2)
array_1 = np.random.rand(1,2)

# Stack across dim=2 
stack = np.stack((array_0, array_1), 2)
print(stack.shape)


# In[33]:


# Concatenating across a dimension

array_0 = np.random.rand(1,2)
array_1 = np.random.rand(1,2)
array_2 = np.random.rand(1,2)

# Concatenate across dim=0 
stack = np.concatenate((array_0, array_1, array_2), 0) # shape=(3,2)
print(stack.shape)

# Concatenate across dim=1
stack = np.concatenate((array_0, array_1, array_2), 1) # shape=(1,6)
print(stack.shape)

# Spliting arrays into multiple sub-arrays
print(" Splitting arrays ")
array = np.random.rand(3,2)
split = np.split(array, 3,0)
print(array)
print(split)


# ### Numpy from text
# Numpy offers handy tools to load data from text files. We will use np.loadtxt() and genfromtxt() to read the numerical data stored inside the data/data_0.txt file.
# When you open "data/data_0.txt" file you will see the following numerical data stored.
# 
# Sample description of the data.
# Value1  Value2  Value3
# 0.6  0.1  0.2
# 0.4  0.4  0.3
# 0.2  0.6  0.9
# 0.1  0.9  0.6

# In[35]:


data_1 = np.loadtxt('data/data_0.txt', skiprows=2)
print(data_1, '\n')
data_2 = np.genfromtxt('data/data_0.txt',skip_header=2)
print('\n', data_2, '\n')


# In[38]:


# Let's multiply the elements of the data array by random integer between 0 and 9
# and save the result as data/data_rand.txt
np.savetxt('./data/data_rand.txt', data_1*np.random.randint(0,9), delimiter=" ")
np.savetxt('./data/data_rand_comma_speparated.txt', data_1*np.random.randint(0,9), delimiter=",")


# # Scipy
# 

# Numpy provides powerful set of tools for manipulating arrays and performing matrix operations. The Scipy library builds upon Numpy to large number of algorithms and mathematical functions. In this class we will be using only few of the features, like ***matrix-inversion, norm computation, reading images from disk etc***. However, Scipy provides a wide range of mathematical operation for **linear algebra, optimization, image analysis, fast fourier transforms, etc**.  
# 
# We recommend to have a look at the documentation of scipy. https://docs.scipy.org/doc/

# In[39]:


import scipy.linalg
# Computing Norm of a vector
vec_a = np.array([[1,2],[3,4]])

# L1 Norm
print(scipy.linalg.norm(vec_a,1))

# Frobenius norm # this is the default norm
print(scipy.linalg.norm(vec_a,"fro"))
print(np.sqrt(np.square(vec_a).sum()))


# In[36]:


# Computing Matrix Inverse
R = np.random.rand(3,3)
R_inv = scipy.linalg.inv(R)
print(np.dot(R, R_inv)) # Identity matrix
R_inv = np.linalg.inv(R)
print(np.dot(R, R_inv)) # Identity matrix


# #### Loading Images
# 
# We will use the matplotlib library for loading, saving and displaying images. 

# In[7]:


import matplotlib.pyplot as plt
import imageio
# Let's load an image with scipy 
img = plt.imread('data/images/lena.jpeg')
print(img.shape) # RGB Image
plt.imshow(img)
plt.show()


# In[54]:


# Displaying data as a heat map

img = plt.imread('./data/depth_map.png')
plt.imshow(img, cmap='gray')
plt.show()


# In[56]:


plt.imshow(img, cmap='winter')
plt.show()
plt.imsave('./data/depth_map_hot.png', img, cmap='hot')


# In[49]:


plt.imshow(img, cmap='hot',  clim=(0.0, 0.9))
plt.show()


# ### Functions
# Functions are created with keyword **def**. The statement section is mandatory while the return statement is optional. 
# 

# <html>
#     <b>def</b> name(arg1, arg2,...,argN):<br>
#    &emsp; statement<br>
#     &emsp; <b> return</b> value<br>
# </html>

# In[38]:


# Let's create a function that does nothing
def do_nothing():
    pass

# Lets write a function that projects a point a 3D from world to camera coordinates
def project_points(X, R, T):
    return np.dot(R, X) + T

# Let create a rotation matrix that rotates points in 3D by 45 degrees across the x-axis

R = np.array([[1, 0, 0],[0, np.cos(45), -1*np.sin(45)],[0, np.sin(45), np.cos(45)]])
T = np.array([[0],[20],[30]])

X_world = np.array([[10],[20],[30]])
X_camera = project_points(X_world, R, T)

print(X_world)
print(X_camera)


# ### Python Classes
# 
# Python supports object oriented programming. Python classes are created using keyword **class**, as follows

# <html>
#     <b>class</b> camera_projection:<br>
#    &emsp; <b>pass</b>
#     <br>
# </html>
# 
# 
# camera_projection is the name of the class. The class has no methods/member functions/ or attributes. 
# 

# In[39]:


class camera_projection:
        pass

cp = camera_projection()
print(cp)

# It is possible to add new attributes to the class 
cp.attribute_1  = 10
print(cp.attribute_1)


# The builtin **\__init\__()** serves as a constructor for Python classes. the method **\__init\__()** is invoked when an instance of the class is created. Let's create a class that has the **\__init\__()** constructor.

# In[40]:


# let's create a class for projecting 3D points from World Coordinate system to the camera coordinate system 
# using known rotation and translation matrices.

class world_to_camera():
    def __init__(self, rot_mat, t_vec):
        self.R = rot_mat
        self.T = t_vec
        
    def proj_w_2_c(self, X):
        return np.dot(self.R, X) + self.T

R = np.array([[1, 0, 0],[0, np.cos(45), -1*np.sin(45)],[0, np.sin(45), np.cos(45)]])
T = np.array([[0],[20],[30]])

projector = world_to_camera(R, T)

X_world = np.array([[10],[20],[30]])
X_camera = projector.proj_w_2_c(X_world)

print(" Rotation matrix \n ", projector.R)
print(" Translation Vec \n ", projector.T)

print("In world coordinates \n", X_world)
print("In camera coordinates \n", X_camera)


# #### Inheritance
# We can extend Python classes via inheritance. Let's create class_a and extend it to class_b which inherits class_a and adds new attributes and member functions.

# In[1]:



class class_a():
    def __init__(self, arg_1):
        self.attr_1 = arg_1
    def method_1(self):
        print("method from class_a", self.attr_1)

class class_b(class_a):
    def __init__(self, arg_1, arg_2):
        super(class_b, self).__init__(arg_1)
        self.attr_2 = arg_2
    def method_2(self):
        print("new method from class_b ", self.attr_2)

obj = class_b(1,2)


# In[3]:


obj.method_1()
obj.method_2()

