from __future__ import print_function

from itertools import cycle
from time import time
import logging
import matplotlib.pyplot as plt
from numpy import interp
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

# from sklearn.neighbors import KNeighborsClassifier
# from  sklearn.metrics import accuracy_score

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# Definition of functions
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=25, resize=0.9)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# Binarize the output
#y = label_binarize(y, classes=lfw_people.target)

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

###############################################################################
# Part (1) - Split into a training set and a test set using a stratified k fold
# split into a training and testing set
#y = y == 9
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))

## Compute mean face
print("Compute mean face")
mean_face = X_train.mean(axis=0).reshape(1, X_train.shape[1])
print("mean_face shape: " + str(mean_face.shape))

###############################################################################
# Part (1) - Perform a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 10
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))

t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))
print("eigenfaces shape: " + str(eigenfaces.shape))

###############################################################################
# Part (2) - Projecting the input data on the eigenfaces orthonormal basis
# output : X_train_pca and X_test_pca
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()

X_train_pca = pca.transform(X_train) # Projection onto Principal Components (coefficients/weights)
print("X_train_pca shape:" + str(X_train_pca.shape))
X_test_pca = pca.transform(X_test)
print("X_test_pca shape:" + str(X_test_pca.shape))

print("done in %0.3fs" % (time() - t0))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#### Face Reconstruction
faceIndex = 24 # 25th face out of 1941 faces in the dataset

face = X_train[faceIndex] # Original face image
faceVector = face.flatten() - mean_face # Face substracted with mean face
output = mean_face.copy() # Output image
eigenvectors = X_train_pca[faceIndex] # eigenvectors returned by PCA

# For each principal component/eigenface
for componentIdx in range(n_components):
    weights = faceVector.dot(eigenvectors[componentIdx]) # Dot product between original face and eigenvectors
    output = output + eigenfaces[componentIdx].flatten() * weights # Sum of the mean face with the eigenfaces multiplied by weights
output = output.reshape(112, 84)

# Display reconstruction result
n_col = 2
n_row = 1
plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

plt.subplot(n_row, n_col, 1)
plt.imshow(face.reshape((h, w)), cmap=plt.cm.gray)
plt.title("Original", size=12)

plt.subplot(n_row, n_col, 2)
plt.imshow(output.reshape((h, w)), cmap=plt.cm.Greys)
plt.title("Face n°" + str(faceIndex) + " Reconstruction", size=12)

plt.xticks(())
plt.yticks(())
plt.show()

#############################################################################
# Part (3) - Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=1000, n_jobs=7)
clf = rfc.fit(X_train_pca, y_train)

###############################################################################
# Part (3) - Nearest Neighbor Classifier
#neigh = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10, metric='manhattan', n_jobs=7))
#clf = neigh.fit(X_train_pca, y_train)
#print("done in %0.3fs" % (time() - t0))

###############################################################################
# Part (3) - Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#clf = GridSearchCV(SVC(kernel='rbf', degree=2, class_weight='balanced', probability=True, verbose=False), param_grid, n_jobs=7)
#clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=False))
#clf = clf.fit(X_train_pca, y_train)

#y_score = clf.decision_function(X_test_pca)

print("done in %0.3fs" % (time() - t0))
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)

###############################################################################
# Part (4) - Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
#print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
disp = plot_confusion_matrix(clf, X_test_pca, y_test, display_labels=target_names, cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title("Confusion Matrix")
plt.show()

# Accuracy
print("Average accuracy %0.3f" % accuracy_score(y_test, y_pred))

# ROC curves
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
#from sklearn.metrics import roc_curve, auc
print("y_test shape: " + str(y_test.shape))
#print("y_score shape: " + str(y_score.shape))
#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
#mean_tpr /= n_classes

#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
#lineWidth = 2
#plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#         label='Courbe ROC micro-moyenne (AUC = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)

#plt.plot(fpr["macro"], tpr["macro"],
#         label='Courbe ROC macro-moyenne (AUC = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)

#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#for i, color in zip(range(22), colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=lineWidth,
#             label='Courbe ROC de {0} (AUC = {1:0.2f})'
#             ''.format(i, roc_auc[i]))

#plt.plot([0, 1], [0, 1], 'k--', lw=lineWidth)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('Taux de Faux Positifs')
#plt.ylabel('Taux de Vrai Positifs')
#plt.title('Eigenfaces : Courbe ROC pour le classifieur SVM')
#plt.legend(loc="lower right")
#plt.show()

###############################################################################
# Part (5) - Qualitative evaluation of the predictions using matplotlib
# plot the result of the prediction on a portion of the test set
print("Most significative eigenfaces.....")
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)
# plot the most significative eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
