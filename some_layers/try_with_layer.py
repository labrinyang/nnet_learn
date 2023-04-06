# Import the necessary layers from a module called 'layers'
from common.layers import MulLayer, AddLayer

# Define the prices and quantities of apples and oranges, as well as the tax rate
apple_price = 100
orange_price = 150
apple_quantity = 2
orange_quantity = 3
tax_rate = 1.1

# Create instances of the necessary layers
multiply_apple_layer = MulLayer()
multiply_orange_layer = MulLayer()
add_layer = AddLayer()
multiply_tax_layer = MulLayer()

# Perform forward propagation
# First layer
apple_subtotal = multiply_apple_layer.forward(apple_price, apple_quantity)
orange_subtotal = multiply_orange_layer.forward(orange_price, orange_quantity)
# Second layer
total_subtotal = add_layer.forward(apple_subtotal, orange_subtotal)
# Third layer
total_price = multiply_tax_layer.forward(total_subtotal, tax_rate)

# Perform backward propagation
dtotal_price = 1
# Third layer
dtotal_subtotal, dtax_rate = multiply_tax_layer.backward(dout=dtotal_price)
# Second layer
dapple_subtotal, dorange_subtotal = add_layer.backward(dout=dtotal_subtotal)
# First layer
dapple_price, dapple_quantity = multiply_apple_layer.backward(dout=dapple_subtotal)
dorange_price, dorange_quantity = multiply_orange_layer.backward(dout=dorange_subtotal)

# Print the total price and gradients of each variable
print(total_price)
print(dapple_quantity, dapple_price, dorange_price, dorange_quantity, dtax_rate)
