from sklearn.linear_model import LinearRegression
import numpy as np

# Our data
sizes = np.array([650, 785, 1200, 720, 975]).reshape((-1, 1))
prices = np.array([772, 998, 1200, 850, 1150])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(sizes, prices)

# Get the size from the user
size_new = float(input("Enter the size of the house in sq ft: "))

# Now we can predict the price of a house of the given size
size_new = np.array([size_new]).reshape((-1, 1))
price_new = model.predict(size_new)

print(f"The predicted price for a house of size {size_new[0][0]} sq ft is {price_new[0]} thousand dollars.")
