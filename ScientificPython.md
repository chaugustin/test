* This is most of the code from Quentin's course on scientific programming in Python

```python

mylist = [1, 2.41341]
mylist.append("We can mix types !")

print(mylist)
print(type(mylist))
print("Length is {} long.\n".format(len(mylist)))

print("There are {} ones in this list.\n".format(mylist.count(1)))

mylist.reverse()
print("Reversed ! {}".format(mylist))
print(1 in mylist)
print(2 in mylist)
for index, value in enumerate(mylist) :
    print("Element number {} in the list has the value {}".format(index, value))
    myint = 2
myfloat = 3.14
print(type(myint), type(myfloat))
def fib(n) :
    if n < 2 :
        return n
    else :
        return fib(n-1) + fib(n-2)
def printFib(i) :
    print("The {}th number of the Fibonnaci sequence is {}.".format(i, fib(i)))
# Read a CSV file, show the first few rows
data = pd.read_csv("sheep.csv")
data.head()
# Show columns of the datafra
print(data.columns, "\n")

# We can access columns by their name directly with a . in between
# Here, we can also use data["ShannonDiversity"][:5], dict-style
print(data.Directionality[:5], "\n")

# Mean of each column
print(data.mean())

# How many entries are in the dataset ?
print(data.count(), "\n")

# The dataset contains some N/A values in the Virulence column. 
# Let's get rid of everything that contains an N/A
print("Before N/A dropping : {} entries".format(len(data)))

data = data.dropna(how="any") # how="all" - remove only if all the data is missing
print("After N/A dropping : {} entries".format(len(data)))

# Let's also get rid of the image ID column, we don't really need it
data.drop("Image", axis=1, inplace=True)
data.head(5)

# Here, we specify that we're dropping a *column* ( along the axis = 1 )
# Let's grab the foci count in sheep with high directionality coefficients
high_directionality = data[data.Directionality < 0.3].FociCount

# and plot it a moving average over ten points
pd.rolling_mean(high_directionality, 10).plot()

# This plot wasn't very meaningful, let's group by individual sheep first.
# Here, we find the mean value of each column for each individual sheep..
vir = sheep.groupby("Sheep").mean()

# Let's compare some of their values in a bar plot.)
vir[["Lacunarity", "Entropy", "Directionality"]].plot(kind="bar", ylim=(0, 0.8))

# This is the main matplotlib import
import matplotlib.pyplot as plt
%matplotlib inline

# In IPython Notebooks, you can use this "magic" command to get figures inline :
# %matplotlib inline
# Spyder users should already have inline plots in IPython kernels

# This magic command ( as defined by the % in front of it ) keeps
# my plots directly below IPython Notebook cells. All magic commands
# are only valid in an IPython environment. 


# The following makes figures bigger and increases label sizes for legibility
mpl.rcParams["figure.figsize"] = (14, 5)
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14

# This cell contains a few optional things.


 ############### coffees #####################
 # Read data
data = pd.read_csv("CoffeeTimeSeries.csv")
data.head(3)

# Time array
time = pd.to_datetime(data.Timestamp, dayfirst=True)

print(time[:3])

# List of weekdays
weekdays = [day.isoweekday() for day in time]

print(weekdays[:20])

# When, during the day, are coffees had ?
data.hist(column="Hour", bins=np.arange(6, 20), align="left", width=1)



# seaborn is a fantastic visualisation library in its own right, but here
# I'm using it as prettifying library for matplotlib, amongst other things.
# Importing it gives me beautiful plots, but it isn't necessary.
import seaborn
# seaborn doesn't come with Anaconda. It is in the Conda repo, however,
# so you can install it from the terminal using either
# conda install seaborn
# or
# pip install seaborn

# Let seaborn thrusters to max awesome
seaborn.set_style("darkgrid")

x = np.random.exponential(3, size=10000)

plt.subplot(1, 2, 1)
plt.plot(x[::50])
plt.axhline(x.mean(), color="red")

plt.subplot(1, 2, 2)
plt.hist(x, 50)
plt.axvline(x.mean(), c="r") # c="r" is shorter, both are valid

plt.show()

# In an interactive setting, you will need to call plt.show()
# In IPython, you can call "%matplotlib inline" after importing and you're good to go.
# From here on, I'll drop plot.show().

# Like Matlab, we can feed style arguments to the plot function
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(scale = 0.2, size = 100)

# Calling plot() several times without creating a new figure puts them on the same one
plt.plot(x, y, "r.")
plt.plot(x, np.sin(x), "k-")

# 100 x 3 matrix on a scatter plot, third dimension plotted as size
x = np.random.uniform(low = 0, high = 1, size = (100, 3))

# Transparency ? Sure.
plt.scatter(x[:, 0], x[:, 1], s=x[:, 2]*500, alpha = 0.4)

# Set axis limits
plt.xlim([0, 1])
plt.ylim([0, 1])

# 100 x 3 matrix on a scatter plot, third dimension plotted as size
x = np.random.uniform(low = 0, high = 1, size = (100, 3))

# Transparency ? Sure.
plt.scatter(x[:, 0], x[:, 1], s=x[:, 2]*500, alpha = 0.4)

# Set axis limits
plt.xlim([0, 1])
plt.ylim([0, 1])# 100 x 3 matrix on a scatter plot, third dimension plotted as size

# Area plots ?
x = np.linspace(0, 6 * np.pi, 300)
y = np.exp(-0.2*x) * np.sin(x)

plt.plot(x, y, "k", linewidth=4)
plt.fill_between(x, y, y2=0, facecolor="red", alpha=0.7)

# Error bars ? We have ten measurements of the above process, the decaying sine wave
noise = np.random.normal(size = (len(y), 10)).T

# Let's add some Gaussian noise to our observations, using broadcasting
measuredy = y + 0.05 * noise

# Let's assume we know our error is Gaussian, for simplicity. Compute mean and std :
estmean = measuredy.mean(axis=0)
eststd = measuredy.std(axis=0)

# Plot the estimated mean with two standard deviation error bars, and the real signal
plt.plot(x, y, "r", lw=3)
plt.errorbar(x, estmean, yerr = eststd * 2, lw=1)
# Reset plotting style
seaborn.set_style("white")

import scipy as sp
sp.__version__

# Two dimensional, or image plots ?
import scipy.ndimage as nd

# Initial 2D noisy signal
X = np.random.normal(size=(256, 256))
plt.figure()
plt.imshow(X, cmap="gray")
plt.title("Original 2D Gaussian noise")

# We'll use a 2D Gaussian for smoothing, with identity as covariance matrix
# We'll grab a scipy function for it for now
plt.figure()
temp = np.zeros_like(X)
temp[128, 128] = 1.
plt.imshow(nd.filters.gaussian_filter(temp, 20), cmap="coolwarm")
plt.title("Gaussian kernel")

# Generate the Perlin noise
perlin = np.zeros_like(X)
for i in 2**np.arange(6) :
    perlin += nd.filters.gaussian_filter(X, int(i), mode="wrap") * i**2
    
# and plot it several ways
plt.figure()
plt.imshow(perlin, cmap="gray")
plt.title("Perlin Noise")

plt.figure()
plt.imshow(perlin, cmap="bone")
plt.contour(perlin, linewidths=3, cmap="jet")
plt.title("Greyscale, with contours")
#plt.xlim([0, 256])
#plt.ylim([0, 256])
#plt.axes().set_aspect('equal', 'datalim')

# And, of course ( stolen from Jake VanderPlas )

def norm(x, x0, sigma):
    return np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)

def sigmoid(x, x0, alpha):
    return 1. / (1. + np.exp(- (x - x0) / alpha))
    
# define the curves
x = np.linspace(0, 1, 100)
y1 = np.sqrt(norm(x, 0.7, 0.05)) + 0.2 * (1.5 - sigmoid(x, 0.8, 0.05))

y2 = 0.2 * norm(x, 0.5, 0.2) + np.sqrt(norm(x, 0.6, 0.05)) + 0.1 * (1 - sigmoid(x, 0.75, 0.05))

y3 = 0.05 + 1.4 * norm(x, 0.85, 0.08)
y3[x > 0.85] = 0.05 + 1.4 * norm(x[x > 0.85], 0.85, 0.3)

with plt.xkcd() :
    
    plt.plot(x, y1, c='gray')
    plt.plot(x, y2, c='blue')
    plt.plot(x, y3, c='red')

    
    plt.text(0.3, -0.1, "Yard")
    plt.text(0.5, -0.1, "Steps")
    plt.text(0.7, -0.1, "Door")
    plt.text(0.9, -0.1, "Inside")
    
    plt.text(0.05, 1.1, "fear that\nthere's\nsomething\nbehind me")
    plt.plot([0.15, 0.2], [1.0, 0.2], '-k', lw=0.5)
    
    plt.text(0.25, 0.8, "forward\nspeed")
    plt.plot([0.32, 0.35], [0.75, 0.35], '-k', lw=0.5)
    
    plt.text(0.9, 0.4, "embarrassment")
    plt.plot([1.0, 0.8], [0.55, 1.05], '-k', lw=0.5)
    
    plt.title("Walking back to my\nfront door at night:")
    
    plt.xlim([0, 1])
    plt.ylim([0, 1.5])
```
