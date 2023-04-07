# 好像还8错
# 总结: 简单看代码行数，普通方法要更少。但是这是因为输入太少了。当面对更多的输入时，将代码变成class或者function会更好。我认为class比起function会更好。
# Summary: For small inputs, conventional methods require fewer lines of code. But as the inputs increase,
#          it's better to use classes or functions. In my opinion, using classes is better than functions.
from common.layers import MulLayer, AddLayer

def compute_total_price_and_gradients(prices, quantities, tax_rate):
    # Create instances of the necessary layers
    multiply_layer = MulLayer()
    add_layer = AddLayer()
    multiply_tax_layer = MulLayer()

    # Perform forward propagation
    subtotals = []
    for price, quantity in zip(prices, quantities):
        subtotal = multiply_layer.forward(price, quantity)
        subtotals.append(subtotal)
    total_subtotal = add_layer.forward(*subtotals)
    total_price = multiply_tax_layer.forward(total_subtotal, tax_rate)

    # Perform backward propagation
    dtotal_price = 1
    dtotal_subtotal, dtax_rate = multiply_tax_layer.backward(dout=dtotal_price)
    dsubtotals = add_layer.backward(dout=dtotal_subtotal)
    dprices = []
    dquantities = []
    for subtotal, dsubtotal in zip(subtotals, dsubtotals):
        dprice, dquantity = multiply_layer.backward(dout=dsubtotal)
        dprices.append(dprice)
        dquantities.append(dquantity)

    return total_price, dprices, dquantities, dtax_rate


# Define the prices and quantities of apples and oranges, as well as the tax rate
apple_price = 100
orange_price = 150
apple_quantity = 2
orange_quantity = 3
tax_rate = 1.1

# Call the function to compute the total price and gradients
total_price, dprices, dquantities, dtax_rate = compute_total_price_and_gradients(
    prices=[apple_price, orange_price],
    quantities=[apple_quantity, orange_quantity],
    tax_rate=tax_rate
)

# Print the total price and gradients of each variable
print(total_price)
print(dquantities[0], dprices[0], dprices[1], dquantities[1], dtax_rate)
