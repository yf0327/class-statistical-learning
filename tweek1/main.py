import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, auc

with open('C:/Users/11208/PycharmProjects/pythonProject4/1/score.csv', newline='') as f:
    reader = csv.reader(f)
    s = list(reader)
tmp = s[0]
score = np.array([float(item) for item in tmp])

with open('C:/Users/11208/PycharmProjects/pythonProject4/1/label.csv', newline='') as f:
    reader = csv.reader(f)
    l = list(reader)
tmp = l[0]
label = np.array([float(item) for item in tmp])
classes=["high","low"]
#predict score
threshold = 0.05
predict = score.copy()
predict[predict>threshold] = 1
predict[predict<threshold] = 0
#calculate confusion matrix, TPR，FPR，Precision，Recall，F1-score，Accuracy
cm = confusion_matrix(label, predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes).plot()
print('Confusion Matrix: \n',cm)
fpr, tpr, thresholds = metrics.roc_curve(label, score)
TPR=cm[0][0]/(cm[0][0]+cm[1][0])
FPR=cm[0][1]/(cm[0][1]+cm[1][1])
print('TruePositiveRate: ', TPR)
print('FalsePositiveRate: ', FPR)
accuracy = accuracy_score(label, predict)
print('Accuracy: ', accuracy)
precision = precision_score(label, predict)
print('Precision: ', precision)
recall = recall_score(label, predict)
print('Recall: ', recall)
f1 = f1_score(label, predict)
print('F1 score: ', f1)


plt.plot(fpr,tpr)
plt.show()



plt.figure()
roc_auc = auc(fpr, tpr)
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()