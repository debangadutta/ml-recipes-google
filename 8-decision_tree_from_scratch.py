'''
Toy dataset.
The last column is the label.
The first two columns are features.
Interesting note: I've written this so the 2nd and 5th examples
have the same features, but different labels - so we can see how the
tree handles this case.
'''
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

# Column labels.
# These are used only to print the tree.
header = ["color", "diameter", "label"]