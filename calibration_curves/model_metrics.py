import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss, multilabel_confusion_matrix
from sklearn import metrics
import gc
import matplotlib.pyplot as plt
import seaborn as sns


def BNN_predict(num_classes, predict_probs, true_labels, model_type):
    
    preds = np.mean(predict_probs, axis=0)
    
    var = np.sum(np.var(predict_probs, axis=0), axis=1)
    
    std = np.sqrt(var)

    if model_type == 'multi_class':
        entropy = -np.sum(preds * np.log(preds + 1E-14), axis=1) 
    else:
        entropy = -(preds * np.log(preds + 1E-14)) - ((1-preds) * np.log((1-preds) + 1E-14))

    # Compute epistemic and aleatoric uncertainty based on kwon et. al. decomposition
    epistemic_kwon = np.sum(np.mean(predict_probs ** 2, axis=0) - np.mean(predict_probs, axis=0) ** 2, axis=1)
    aleatoric_kwon = np.sum(np.mean(predict_probs * (1 - predict_probs), axis=0), axis=1)

    # Compute epistemic and aleatoric uncertainty based on depeweg et. al. decomposition
    if model_type == 'multi_class':
        aleatoric_depeweg = np.mean(-np.sum(predict_probs * np.log(predict_probs + 1E-14), axis=2), axis=0)
    else:
        aleatoric_depeweg = np.mean(-(predict_probs * np.log(predict_probs + 1E-14)) - (1-predict_probs) * np.log((1-predict_probs) + 1E-14), axis=0)
    
    epistemic_depeweg = entropy - aleatoric_depeweg

    # Model NLL Calculation depends on model_type
    if(model_type == 'multi_class'):
        nll = multiclass_NLL(true_labels, preds)
    else:
        nll = multilabel_NLL(true_labels, preds)
        epistemic_depeweg = np.sum(epistemic_depeweg, axis=1)
        aleatoric_depeweg = np.sum(aleatoric_depeweg, axis=1)
        entropy = np.sum(entropy, axis=1)
    
    # Normalized entropy for number of classes
    if(model_type == 'multi_class'):
        norm_entropy = entropy/np.log2(num_classes)
    else:
        norm_entropy = entropy/np.log2(2**num_classes)

    return preds, entropy, nll, std, var, norm_entropy, epistemic_kwon, aleatoric_kwon, epistemic_depeweg, aleatoric_depeweg



def multiclass_metrics(true_labels, pred_labels, class_labels):
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(str(conf_matrix))

    class_report = classification_report(true_labels, pred_labels, labels=[0,1,2,3,4], target_names=class_labels)
    print(class_report)

    accuracy = accuracy_score(true_labels, pred_labels)
    print("Accuracy: %.4f" % accuracy)

def multilabel_metrics(true_labels, pred_labels, class_labels):
    print("Confusion Matrix:")
    conf_matrix = multilabel_confusion_matrix(true_labels, pred_labels)
    print(str(conf_matrix))

    class_report = classification_report(true_labels, pred_labels, labels=[0,1,2,3,4], target_names=class_labels)
    print(class_report)

    hloss = hamming_loss(true_labels, pred_labels)
    print("Hamming loss: %.3f" % hloss)

    accuracy = accuracy_score(true_labels, pred_labels)
    print("Accuracy: %.4f" % accuracy)

def binary_metrics(true_labels, pred_labels, class_labels):
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(str(conf_matrix))

    class_report = classification_report(true_labels, pred_labels, labels=[0,1], target_names=class_labels)
    print(class_report)

    hloss = hamming_loss(true_labels, pred_labels)
    print("Hamming loss: %.3f" % hloss)

    accuracy = accuracy_score(true_labels, pred_labels)
    print("Accuracy: %.4f" % accuracy)

def multiclass_NLL(true_labels, model_preds):
    return np.sum(-(true_labels * np.log(model_preds)), axis=1)

def multilabel_NLL(true_labels, model_preds):
    if(len(true_labels.shape) == 1):
        true_labels = np.expand_dims(true_labels, 1)
    
    return -np.sum((true_labels * np.log(model_preds)) + ((1-true_labels) * np.log(1-model_preds)), axis=1)

def plot_ratio_data_retained(uncertainty_measure, y_pred, y_true, uncertainty_label, metric=accuracy_score):
    if metric == accuracy_score:
        metric_label = 'Test Accuracy'
    elif metric == metrics.f1_score:
        metric_label = 'Test F1'
    elif metric == metrics.recall_score:
        metric_label = 'Recall'
    elif metric == metrics.precision_score:
        metric_label = 'Precision'
    elif metric == metrics.matthews_corrcoef:
        metric_label = 'Matthews Correlation Coeff'
    else:
        print("Metric Not Implemented")
        return

    acc_cal_noe = []
    one_pct = int(y_true.shape[0] * .01)

    pred_labels_sort = [v for _,v in sorted(zip(uncertainty_measure,y_pred), key = lambda x: x[0], reverse=True)]
    true_labels_sort = [v for _,v in sorted(zip(uncertainty_measure,y_true), key = lambda x: x[0], reverse=True)]

    for p in range(100):
        tl = true_labels_sort[p*one_pct:]
        pl = pred_labels_sort[p*one_pct:]
        if metric != metrics.matthews_corrcoef:
            accuracy=metric(tl,pl, zero_division=1.0)
        else:
            accuracy=metric(tl,pl)
        acc_cal_noe.append(accuracy)

    acc_cal_noe.reverse()
    
    plot = sns.lineplot(acc_cal_noe)
    plot.set(xlabel=f"Ratio of Data Retained ({uncertainty_label})", ylabel=metric_label, title=f"{metric_label} vs Ratio of Data Retained")
    plt.show()
    plt.close()

def plot_ratio_data_retained_comparison(uncertainty_measure, y_pred, y_true, uncertainty_label, model_labels,title="Test Accuracy vs Ratio of Data Retained"):
    acc_cal_dict = {}
    one_pct = int(y_true[0].shape[0] * .01)

    for index, model in enumerate(model_labels):
        acc_cal_noe = []

        pred_labels_sort = [v for _,v in sorted(zip(uncertainty_measure[index],y_pred[index]), key = lambda x: x[0], reverse=True)]
        true_labels_sort = [v for _,v in sorted(zip(uncertainty_measure[index],y_true[index]), key = lambda x: x[0], reverse=True)]

        for p in range(100):
            tl = true_labels_sort[p*one_pct:]
            pl = pred_labels_sort[p*one_pct:]
            accuracy=accuracy_score(tl,pl)
            acc_cal_noe.append(accuracy)
        acc_cal_noe.reverse()
        acc_cal_dict[model] = acc_cal_noe

    plot = sns.relplot(data=acc_cal_dict, kind="line",height=4,aspect=1.7)
    plot.set(xlabel=f"Ratio of Data Retained ({uncertainty_label})", ylabel="Test Accuracy", title=title)
    legend = plot.legend.remove()
    plt.legend(loc='best')
    
    sns.despine(top=False, right=False, left=False, bottom=False)
    plt.show()
    plt.close()

def plot_train_curves(df):    
    acc_results = df[['Accuracy','val_Accuracy']]
    loss_results = df[['Loss', 'val_Loss']]
    
    fig, ax = plt.subplots(2,1, figsize=(8,10))
    
    sns.lineplot(acc_results, ax=ax[0], dashes=False)
    sns.lineplot(loss_results, ax=ax[1], dashes=False)
    
    ax[0].set(xlabel='Epoch', ylabel='Accuracy')
    ax[1].set(xlabel='Epoch', ylabel='Loss')

    plt.show()
    plt.close()
