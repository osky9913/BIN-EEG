import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score #works
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.metrics import det_curve

label_names = ['HandStart','FirstDigitTouch','BothStartLoadPhase','LiftOff','Replace','BothReleased']


stats = {}
fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))


for i in range(12):
    print(i)



# Load the data from the file
    with open('validationsIndividuals/'+str(i)+'.txt', 'r') as f:
        data = json.load(f)

    y_true = np.array([d['targets'] for d in data],dtype=int)
    y_score = np.array([d['outputs'] for d in data])
    #predicted = np.array([d['predicted'] for d in data])


    precision, recall, thresholds = precision_recall_curve(y_true.reshape(-1), y_score.reshape(-1))
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_score >= best_threshold).astype(int)


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true.reshape(-1), y_score.reshape(-1))
    
    met = {}

    met["accuracy"] =accuracy
    met["precision"] =precision
    met["recall"] =recall
    met["f1"] =f1
    met["auc"] =auc
    stats[str(i)]= met

    #print(classification_report(y_true, y_pred,target_names=label_names))


    """
    cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(np.arange(6), ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased'], rotation=45)
    plt.yticks(np.arange(6), ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased'])
    plt.show()

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
    plt.plot(fpr, tpr)
    plt.title('ROC Curve (AUC = %0.2f)' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


    fpr, tpr, thresholds = det_curve(y_true.reshape(-1), y_score.reshape(-1))

    # Plot the DET curve
    plt.plot(fpr, tpr, linewidth=2)
    plt.xscale('log')
    plt.xlim([0.001, 1])
    plt.ylim([0.001, 1])
    plt.xlabel('False Alarm Rate (FAR)')
    plt.ylabel('Miss Rate')
    plt.title('DET Curve')
    plt.grid(True)
    plt.show()
    #Counter(y_true)
    """

    #threshold = 0.
    #predicted = np.zeros(outputs.shape, dtype=float)
    #predicted[outputs >= threshold] = 1


    """
    totalPrecision= 0
    print("micro: {:.2f}".format(sklearn.metrics.average_precision_score(y_true, y_scores, average='micro')))
    print("macro: {:.2f} ".format( sklearn.metrics.average_precision_score(y_true, y_scores, average='macro')))
    print("weighted: {:.2f} ".format( sklearn.metrics.average_precision_score(y_true, y_scores, average='weighted')))
    #print("samples: {:.2f} ".format( sklearn.metrics.average_precision_score(y_true, y_scores, average='samples')))  
    """



    DetCurveDisplay.from_predictions(y_true=y_true.reshape(-1), y_pred=y_score.reshape(-1),ax=ax_det, name= str(i) )
    RocCurveDisplay.from_predictions(y_true=y_true.reshape(-1), y_pred=y_score.reshape(-1),ax=ax_roc, name= str(i))
    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_det.set_title("Detection Error Tradeoff (DET) curves")

    ax_roc.grid(linestyle="--")
    ax_det.grid(linestyle="--")
plt.legend()
plt.show()
print(stats)

#print(classification_report(targets, predicted,target_names=label_names))

#print(targets)

#false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(targets.ravel(), outputs())
#RocCurveDisplay.from_predictions(targets, outputs)
#plt.show()







# Compute the false positive rate and true positive rate at various thresholds

# Compute the detection error rate (DER) at various thresholds
