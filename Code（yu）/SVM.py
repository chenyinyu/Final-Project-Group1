import sys
import numpy as np
import pickle
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# Save all the Print Statements in a Log file.
old_stdout = sys.stdout
log_file = open("summary_SVM_svc.log","w")
sys.stdout = log_file
X_train = np.genfromtxt('train_img.csv',
                        dtype=int, delimiter=',')
y_train = np.genfromtxt('train_labels.csv',
                        dtype=int, delimiter=',')
X_test = np.genfromtxt('test_img.csv',
                       dtype=int, delimiter=',')
y_test = np.genfromtxt('test_labels.csv',
                       dtype=int, delimiter=',')

img_train= X_train
labels_train = y_train
train_img = np.array(img_train)
train_labels = np.array(labels_train)

img_test=X_test
labels_test =y_test
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
X = train_img

#Labels
y = train_labels

# Prepare Classifier Training and Testing Data
print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.1, random_state = 0)


# Pickle the Classifier for Future Use
print('\nSVM Classifier with gamma = auto; Kernel = polynomial')
print('\nPickling the Classifier for Future Use...')
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=10, gamma='auto', kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.05)
clf.fit(X_train,y_train)

with open('SVM.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('SVM.pickle','rb')
clf = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_test,y_test)

print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_test)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)

print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test,y_pred)

print('\nSVM Trained Classifier Accuracy: ',acc)
print('\nPredicted Values: ',y_pred)
print('\nAccuracy of Classifier on Validation Images: ',accuracy)
print('\nConfusion Matrix: \n',conf_mat)

# Plot Confusion Matrix Data as a Matrix
plt.matshow(conf_mat)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


print('\nMaking Predictions on Test Input Images...')
test_labels_pred = clf.predict(test_img)

print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
acc = accuracy_score(test_labels,test_labels_pred)

print('\n Creating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

print('\nPredicted Labels for Test Images: ',test_labels_pred)
print('\nAccuracy of Classifier on Test Images: ',acc)
print('\nConfusion Matrix for Test Data: \n',conf_mat_test)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
#plt.axis('off')
plt.show()

sys.stdout = old_stdout
log_file.close()
