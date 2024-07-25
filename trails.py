# import numpy as np

# # Step 1: Define custom addition and multiplication functions
# def custom_add(x, y):
#     # Replace this with your custom addition logic
#     return x + y + 1  # Example: adding 1 to the sum

# def custom_multiply(x, y):
#     # Replace this with your custom multiplication logic
#     return x * y + 2  # Example: adding 2 to the product

# # Step 2: Vectorize the custom functions
# vectorized_add = np.vectorize(custom_add)
# vectorized_multiply = np.vectorize(custom_multiply)

# # Step 3: Create some example arrays
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])

# # Step 4: Use the vectorized functions to perform array operations
# custom_sum = vectorized_add(a, b)
# custom_product = vectorized_multiply(a, b)

# print("Custom Sum:", custom_sum)
# print("Custom Product:", custom_product)
# print("Original Sum:", a + b)
# print("Original Product:", a * b)
