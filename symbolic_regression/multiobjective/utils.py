from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)

from symbolic_regression.Program import Program


def create_binary_classification_report(program: Program, 
                                        data: pd.DataFrame, val_data: pd.DataFrame,
                                        target: str, features: List[str], weights: str):
    
    
    pred_train = program.to_logistic(inplace=False).evaluate(data[features])
    pred_train_siz = pd.Series(np.where(pred_train > .5, 1, 0)).astype(int)
    if val_data is not None:
        pred_test = program.to_logistic(inplace=False).evaluate(val_data[features])
        pred_test_siz = pd.Series(np.where(pred_test > .5, 1, 0)).astype(int)

    #################### AUROC ####################
    plt.figure()

    lw = 2

    fpr, tpr, thresholds = roc_curve(
        y_true=data[target], y_score=pred_train, sample_weight=data[weights] if weights else None)
    roc_auc_train = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkgreen',
            lw=lw, label=f'Train ROC curve (area = {roc_auc_train:.3f})')

    if val_data is not None:
        # Calculate the AUROC curve and plot it with matplotlib
        fpr, tpr, thresholds = roc_curve(y_true=val_data[target], y_score=pred_test)
        roc_auc_test = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label=f'Test ROC curve (area = {roc_auc_test:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.grid()
    plt.legend()
    plt.title('AUROC Curve')

    plt.show()

    #################### PRC ####################
    plt.figure()

    lw = 2

    # Calculate precision-recall curve for the training data
    precision_train, recall_train, _ = precision_recall_curve(
        y_true=data[target], probas_pred=pred_train, sample_weight=data[weights] if weights else None)
    pr_auc_train = auc(recall_train, precision_train)
    plt.plot(recall_train, precision_train, color='darkgreen',
            lw=lw, label=f'Train Precision-Recall curve (area = {pr_auc_train:.3f})')

    if val_data is not None:
        # Calculate precision-recall curve for the test data
        precision_test, recall_test, _ = precision_recall_curve(
            y_true=val_data[target], probas_pred=pred_test)
        pr_auc_test = auc(recall_test, precision_test)
        plt.plot(recall_test, precision_test, color='darkorange',
                lw=lw, label=f'Test Precision-Recall curve (area = {pr_auc_test:.3f})')

    plt.grid()
    plt.legend()

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.show()

    cm_train = confusion_matrix(data[target], pred_train_siz)
    print(f"Train Confusion Matrix: \n\n{cm_train}\n")

    if val_data is not None:
        cm_test = confusion_matrix(val_data[target], pred_test_siz)
        print(f"Test Confusion Matrix: \n\n{cm_test}\n")
        print()

    TP_train = cm_train[0][0]
    TN_train = cm_train[1][1]
    FP_train = cm_train[0][1]
    FN_train = cm_train[1][0]

    print(f"Train Accuracy: {accuracy_score(data[target], pred_train_siz):.3f}")
    # print(f"Train Precision Score: {precision_score(data[target], pred_train_siz):.3f}")
    print(f"Train Precision: {TP_train / (TP_train + FP_train):.3f}")

    print(f"Train Recall (Sensitivity): {recall_score(data[target], pred_train_siz):.3f}")
    print(f"Train Specificity: {TN_train / (TN_train + FP_train):.3f}")
    print(f"Train F1: {f1_score(data[target], pred_train_siz):.3f}")
    print(f"Train AUC: {roc_auc_train:.3f}")
    print()
    
    if val_data is not None:
        TP_test = cm_test[0][0]
        TN_test = cm_test[1][1]
        FP_test = cm_test[0][1]
        FN_test = cm_test[1][0]
        print(f"Test Accuracy: {accuracy_score(val_data[target], pred_test_siz):.3f}")
        # print(f"Test Precision Score: {precision_score(val_data[target], pred_test_siz):.3f}")
        print(f"Test Precision: {TP_test / (TP_test + FP_test):.3f}")

        print(f"Test Recall (Sensitivity): {recall_score(val_data[target], pred_test_siz):.3f}")
        print(f"Test Specificity: {TN_test / (TN_test + FP_test):.3f}")
        print(f"Test F1: {f1_score(val_data[target], pred_test_siz):.3f}")
        print(f"Test AUC: {roc_auc_test:.3f}")
        print()