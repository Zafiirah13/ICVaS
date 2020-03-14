import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix,balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score
import itertools
from scipy import interp
from itertools import cycle, islice
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})
figSize  = (12, 8)
fontSize = 20


# ----------------------------------------------------------------------------------
#                             Receiver-Operational Curve
# ----------------------------------------------------------------------------------

def plot_ROC_curve(X_test, y_test, nClasses, fit_model,plots_dir,classes_types):

    le = LabelEncoder()
    labels = le.fit_transform(y_test)
    yTest = np_utils.to_categorical(labels, nClasses)
    preds = fit_model.predict_proba(X_test)


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nClasses):
        fpr[i], tpr[i], _ = roc_curve(yTest[:,i], preds[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nClasses)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nClasses):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nClasses
    
    fpr["micro"], tpr["micro"], _ = roc_curve(yTest.ravel(),preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(12,8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontSize)
    plt.ylabel('True Positive Rate',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    plt.savefig(plots_dir+'_ROC.pdf',bbox_inches = 'tight',pad_inches = 0.1)
    plt.show()
    
    plt.figure(figsize=(12,8))
    colors = cycle(['b', 'darkorange', 'gold', 'purple', 'y', 'brown', 'pink','cornflowerblue', 'olive','green', 'r', ])

    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nClasses), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.0,
                 label='{0} (area = {1:0.2f})'
                 ''.format(classes_types[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontSize)
    plt.ylabel('True Positive Rate',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    
    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    plt.savefig(plots_dir+'_all_ROC.pdf',bbox_inches = 'tight',pad_inches = 0.1)
    plt.show()
    
    return fpr,tpr,roc_auc

# ----------------------------------------------------------------------------------
#                             Confusion Matrix
# ----------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes_types,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    

    print(cm)
    plt.figure(figsize=(9,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cb=plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(classes_types))
    plt.xticks(tick_marks, classes_types, rotation=45)
    plt.yticks(tick_marks, classes_types)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if (cm[i, j] < 0.01) or (cm[i,j] >= 0.75)  else "black",fontsize=18)

    
    plt.ylabel('True label',fontsize = 16)
    plt.xlabel('Predicted label', fontsize = 16)
    plt.tight_layout()


def plot(conf_mat, classes_types, classifier_model, plot_title, X_test, y_test, nClasses,cmap=plt.cm.Reds):

    plt.figure(figsize=(8,6))
    
    plot_confusion_matrix(conf_mat, classes_types, normalize=True, title='Confusion matrix for ' + str(classifier_model) )
    plt.savefig(plot_title +'_CM.pdf',bbox_inches = 'tight',pad_inches = 0.1)
    plt.close()

