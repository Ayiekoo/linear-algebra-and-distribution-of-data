#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Linear algebra

"""
import linear algebra functions in ML;
1. Notation
2. Operations// addition, multiplication etc
3. Matrix factorization // includes singular value decomposition & QR decomposition
"""



# In[21]:


import numpy as np

# Sample data
grades = np.array([[80, 90, 85],
                   [70, 75, 80],
                   [90, 85, 95],
                   [85, 80, 75],
                   [95, 90, 80]])


# In[22]:


# Calculate average grade for each subject
subject_average = np.mean(grades, axis=0)
print("Subject Average:", subject_average)

# Calculate total grade for each student
student_total = np.sum(grades, axis=1)
print("Student Total Grade:", student_total)

# Calculate overall class average
class_average = np.mean(grades)
print("Class Average:", class_average)


# In[24]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
grades_reduced = pca.fit_transform(grades)

# Plot the reduced data
plt.scatter(grades_reduced[:, 0], grades_reduced[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Student Grades")
plt.show()


# In[ ]:


### scalars and vectors
"""
vectors; speed & velocity
scalars; 
"""


# In[27]:


temperatures = [20, 22, 21, 25, 24, 23, 22, 21, 20, 19, 18, 20, 21, 23, 25, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 18, 20]
print(temperatures)


# In[25]:


stock_prices = [100.50, 102.75, 101.80, 103.20, 104.10, 102.95, 100.80]


# In[29]:


exam_scores = [85, 90, 92, 78, 80, 88, 95, 83, 87, 91, 84, 89]

print(exam_scores)


# In[31]:


heights = [165, 170, 172, 168, 175, 160, 165, 178, 173, 169, 171, 166]
print(heights)


# In[33]:


product_prices = [29.99, 14.99, 49.99, 9.99, 39.99, 19.99, 59.99]
print(product_prices)


# In[34]:


import matplotlib.pyplot as plt

# Example 1: Daily Temperature
plt.plot(temperatures)
plt.xlabel('Day')
plt.ylabel('Temperature (Celsius)')
plt.title('Daily Temperature Variation')
plt.show()

# Example 2: Stock Prices
plt.plot(stock_prices)
plt.xlabel('Day')
plt.ylabel('Price (USD)')
plt.title('Stock Prices Variation')
plt.show()

# Example 3: Exam Scores
plt.hist(exam_scores, bins=10)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Exam Scores Distribution')
plt.show()

# Example 4: Heights
plt.scatter(range(len(heights)), heights)
plt.xlabel('Person')
plt.ylabel('Height (cm)')
plt.title('Heights of Individuals')
plt.show()

# Example 5: Product Prices
plt.bar(range(len(product_prices)), product_prices)
plt.xlabel('Product')
plt.ylabel('Price (USD)')
plt.title('Product Prices')
plt.show()


# In[ ]:


### matrix operations
"""
include matrix addition
scalar multiplication
subtraction
transpose of a metrix
rank of a matrix
"""


# In[1]:


####  poison distribution
import numpy as np

# Parameters
n = 100  # Number of trials
p = 0.3  # Probability of success

# Generate random dataset
dataset = np.random.binomial(n, p, size=1000)

# Print the first 10 values in the dataset
print(dataset[:10])


# In[2]:


import matplotlib.pyplot as plt

# Plot histogram
plt.hist(dataset, bins=range(n + 2), align='left', rwidth=0.8)

# Set labels and title
plt.xlabel('Number of Successes')
plt.ylabel('Frequency')
plt.title('Binomial Distribution')

# Show the plot
plt.show()

"""
In this code, we import the numpy library as np. We set the parameters n and p, which represent the number of trials and the probability of success, respectively. Using the np.random.binomial function, we generate a dataset of size 1000 that follows a binomial distribution with the given parameters.

Finally, we print the dataset to observe the generated values. You can adjust the parameters n, p, and the size of the dataset (size) according to your requirements.
"""


# In[ ]:


### probability distribution; poison distribution

"""
poison distribution measures teh probability of an event
occuring over specified period
"""

"""
PROBLEM STATEMENT:

Suppose you are going for a long drive. 
The rate of occurance of good restaurants in a range of 10 miles is 2.
In other words, the mean number of occurance of restaurants in a range of 10 minles is 2
What is the probability that 0, 1, 2, 3, 4, or 5 restaurants will show up in the next 10 miles?
"""


# In[15]:


## import the libraries
from scipy.stats import poisson
import matplotlib.pyplot as plt


# In[17]:


# random variable representing number of restaurants
# mean number of occurances of restaurants in 10 miles

x = [0, 1, 2, 3, 4, 5]
lmbda = 2 

"""
 if λ = 2, it means that the average rate of the Poisson process is 2 events occurring per unit of time or space.
 that is, the average rate of the poisson process is 2 restaurants per unit time or space
"""


# In[18]:


poisson_pd = poisson.pmf(x, lmbda)


# In[19]:


## plot a graph of probability values

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(x, poisson_pd, 'bo', ms=8, label='poisson pmf')
plt.ylabel("Probability", fontsize="18")
plt.xlabel("X = No. of Restaurants", fontsize="18")
plt.title("Poisson Distribution - No. of Restaurants vs Probability", fontsize="18")
ax.vlines(x, 0, poisson_pd, colors='b', lw=5, alpha=0.5)

"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)): This line creates a figure object (fig) and an axes object (ax) using the subplots function from Matplotlib. The 1, 1 argument indicates that we want a single subplot, and figsize=(8, 6) sets the size of the figure to be 8 inches wide and 6 inches tall.

    ax.plot(x, poisson_pd, 'bo', ms=8, label='poisson pmf'): This line plots the data points defined by the x values and poisson_pd values. The 'bo' argument specifies that blue circles should be used to represent the points, ms=8 sets the marker size to 8, and label='poisson pmf' provides a label for the plot.

    plt.ylabel("Probability", fontsize="18"): This line sets the y-axis label of the plot to "Probability" with a font size of 18.

    plt.xlabel("X = No. of Restaurants", fontsize="18"): This line sets the x-axis label of the plot to "X = No. of Restaurants" with a font size of 18.

    plt.title("Poisson Distribution - No. of Restaurants vs Probability", fontsize="18"): This line sets the title of the plot to "Poisson Distribution - No. of Restaurants vs Probability" with a font size of 18.

    ax.vlines(x, 0, poisson_pd, colors='b', lw=5, alpha=0.5): This line adds vertical lines to the plot at the x values, starting from the baseline (0) and extending up to the corresponding poisson_pd values. The colors='b' argument sets the color of the lines to blue, lw=5 sets the line width to 5, and alpha=0.5 sets the transparency of the lines to 0.5.

Overall, this code generates a plot of the Poisson distribution, showing the probability mass function (pmf) as blue circles and vertical lines indicating the probabilities at each point. The labels and title provide additional information about the distribution and the plotted values.

"""


# In[3]:


### binomial distribution 

### another examples

import numpy as np

# Set the parameters
n = 10  # Number of trials
p = 0.5  # Probability of success

# Generate the dataset
dataset = np.random.binomial(n, p, size=1000)

# Print the dataset
print(dataset)


# In[4]:


import matplotlib.pyplot as plt

# Plot the histogram
plt.hist(dataset, bins='auto')

# Set labels and title
plt.xlabel('Number of Successes')
plt.ylabel('Frequency')
plt.title('Binomial Distribution')

# Show the plot
plt.show()

"""
In this code, we import the matplotlib.pyplot module as plt. We then use the plt.hist function to create a histogram of the dataset. The argument bins='auto' automatically determines the appropriate number of bins based on the data.

Next, we set the labels for the x and y axes using plt.xlabel and plt.ylabel, respectively. We also set a title for the plot using plt.title.

Finally, we call plt.show() to display the plot.

You can customize the plot further by modifying various parameters of the hist function or using other plotting functions from matplotlib or other libraries to visualize the data in different ways.
"""


# In[ ]:


### PROBLEM STATEMENT

"""
CONSIDER A RANBDOM EXPERIMENT OF TOSSING A BIASED COIN 6 TIMES WHERE THE POSSIBILITY OF GETTING A HEAD IS 0.6. IF GETTING A HEAD IS CONSIDERED AS A "AUCCESS", THEN THE BINOMIAL DISTRIBUTION TABLE WILL CONTAIN THE POSSIBILITY OF R SUCCESS FOR EACH POSSIBLE VALUE OF R
CALUCLATE THE BINOMIAL DISTRIBUTION
"""


# In[5]:


from scipy.stats import binom
import matplotlib.pyplot as plt


# In[ ]:


### set n and p values
n = 6
p = 0.6


# In[6]:


## define the list of r values
r_values = list(range(n + 1))


# In[7]:


### obtain the mean and variance
mean, var = binom.stats(n, p)


# In[8]:


### list of pmf vaues
dist = [binom.pmf(r, n, p) for r in r_values ]


# In[11]:


### print the table
print("r\tp(r)")
for i in range(n + 1):
    print(str(r_values[i]) + "\t" + str(dist[i]))


# In[12]:


### plot the graph
plt.bar(r_values, dist)
plt.show()

"""
by use of a bar chart
a normal distriution is obtained in the bar chart below
"""


# In[20]:


## uniform distribution
### Bernouli distribution
#### cumulative normal distribution
###### central limit theorem
######## Bayes theorem///bayes law

#### estimation theory


# In[ ]:




