from sklearn import tree

#collect training data
features = [        #(0-> bumpy, 1-> smooth)
    [140, 1],
    [130, 1],
    [150, 0],
    [170, 0]
    ]

labels = [      #(0-> apple, 1-> orange)
    0,
    0,
    1,
    1
    ]

#train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#make predictions
print(clf.predict([[150, 0]]))  #heavy, bumpy
print(clf.predict([[150, 1]]))  #heavy, smooth
print(clf.predict([[100, 0]]))  #light, bumpy
print(clf.predict([[100, 1]]))  #light, smooth