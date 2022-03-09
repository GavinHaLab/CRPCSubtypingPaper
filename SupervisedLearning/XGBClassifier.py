#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.0, 07/20/2021

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from collections import Counter
from xgboost import XGBClassifier, XGBRegressor
from sklearn import preprocessing, ensemble, linear_model, metrics, decomposition
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold

# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from mlens.ensemble import SuperLearner

targets = ['Healthy', 'ARPC', 'Luminal', 'NEPC', 'Basal', 'Patient', 'NonDiff', 'Dual', 'ARlow']
colors = ['#009988', '#0077BB', '#33BBEE', '#CC3311', '#EE7733', '#EE3377', '#BBBBBB', '#9370DB', '#77BB00']
palette = {targets[i]: colors[i] for i in range(len(targets))}
sns.set(font_scale=1.5)
sns.set_style('ticks')


# # create a list of base-models
# def get_models():
#     models = list()
#     models.append(LogisticRegression(solver='liblinear'))
#     models.append(DecisionTreeClassifier())
#     models.append(SVC(gamma='scale', probability=True))
#     models.append(GaussianNB())
#     models.append(KNeighborsClassifier())
#     models.append(AdaBoostClassifier())
#     models.append(BaggingClassifier(n_estimators=10))
#     models.append(RandomForestClassifier(n_estimators=10))
#     models.append(ExtraTreesClassifier(n_estimators=10))
#     return models
#
#
# # create the super learner
# def get_super_learner(x):
#     ensemble = SuperLearner(scorer=accuracy_score, folds=5, shuffle=True, sample_size=x)
#     # add base models
#     models = get_models()
#     ensemble.add(models)
#     # add the meta model
#     ensemble.add_meta(LogisticRegression(solver='lbfgs'))
#     return ensemble


def get_model(model_name, sample_size=None):
    if model_name == 'GradientBoostingClassifier':
        print("Building GradientBoostingClassifier model . . .")
        model = ensemble.GradientBoostingClassifier()
        # define hyper parameter combinations to try
        param_dic = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
                     'n_estimators': [10, 100, 250, 500],
                     'max_depth': [2, 3, 4, 5, None],
                     'min_samples_split': [2, 3, 4],
                     'min_samples_leaf': [1, 3, 5],
                     'max_features': ['sqrt'],
                     'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}
        random_search = RandomizedSearchCV(model, param_distributions=param_dic, n_iter=1000, scoring="accuracy")
        return random_search
    elif model_name == 'RandomForestClassifier':
        print("Building RandomForestClassifier model . . .")
        model = ensemble.RandomForestClassifier()
        param_dic = {'n_estimators': [10, 100, 250, 500],
                     'max_depth': [2, 3, 4, 5, None],
                     'min_samples_split': [2, 3, 4],
                     'min_samples_leaf': [1, 3, 5],
                     'max_features': ['sqrt'],
                     'bootstrap': [True]}
        random_search = RandomizedSearchCV(model, param_distributions=param_dic, n_iter=1000, scoring="accuracy")
        return model
    elif model_name == 'LogisticRegression':
        print("Building LogisticRegression model . . .")
        model = linear_model.LogisticRegression(class_weight='balanced')
        param_dic = {'class_weight': ['balanced'],
                     'solver': ['liblinear'],
                     'C': [100, 10, 1.0, 0.1, 0.01]}
        grid_search = GridSearchCV(model, param_grid=param_dic, scoring="accuracy")
        return grid_search
    elif model_name == 'SVM':
        print("Building SVM model . . .")
        model = SVC()
        param_dic = {'kernel': ['poly', 'rbf', 'sigmoid'],
                     'C': [100, 10, 1.0, 0.1, 0.01]}
        grid_search = GridSearchCV(model, param_grid=param_dic, scoring="accuracy")
        return grid_search
    elif model_name == 'XGBoost':
        print("Building XGBoost model . . .")
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        # model = XGBClassifier()
        # param_dic = {'n_estimators': [100, 1000, 1600, 5000],
        #              'use_label_encoder': [False],
        #              'max_depth': [3, 4, 5, 6],  # 3, 4, 5, 6, 8, 10
        #              'eta': [0.01, 0.1, 0.2],
        #              'subsample': [0.5, 0.6, 0.8, 1.0],
        #              'eval_metric': ['logloss']}
        # grid_search = RandomizedSearchCV(model, param_distributions=param_dic, n_iter=10, scoring="accuracy", cv=2)
        # return grid_search
        return model
    elif model_name == 'XGBRegressor':
        print("Building XGBoost model . . .")
        model = XGBRegressor()
        return model
    elif model_name == 'XGBCategorical':
        print("Building XGBoost model . . .")
        model = XGBClassifier(tree_method="gpu_hist", enable_categorical=True, use_label_encoder=False)
        return model
    # elif model_name == 'SuperLearner':
    #     print("Building SuperLearner model . . .")
    #     model = get_super_learner(sample_size)
    #     return model
    else:
        print("No implemented model specified!")
        return None


def evaluate_model(df, model, path, group_level, model_type, test):
    if model_type == 'XGBRegressor':  # regression build
        # extract sample info and truth values:
        y = df['CCP-score'].values
        # samples = df.index.values
        # CV parameters (set to use minimum of 2 samples per subtype):
        repeats = 100
        folds = 5
        cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
        print("Conducting " + str(folds) + "-fold CV with " + str(repeats) + " repeats.")
        if group_level:
            categories = list(set([item.split('_')[1] for item in list(df.columns) if '_' in item]))
            features = list(set([item.split('_')[2] for item in list(df.columns) if '_' in item]))
            print(" . . . on " + str(len(categories)) + " categories.")
            for category in categories:
                final_r2, final_rmse, final_mae, final_features = [], [], [], []
                for feature in features:
                    feature_name = category + '_' + feature
                    sub_df = df.filter(regex=feature_name)
                    x = sub_df.values
                    if x.size == 0: continue
                    r2, rmse, mae = [], [], []
                    for train, test in cv.split(x, y):
                        x_train, x_test = x[train], x[test]
                        y_train, y_test = y[train], y[test]
                        model.fit(x_train, y_train)
                        prediction = model.predict(x_test)
                        r2.append(metrics.r2_score(y_test, prediction))
                        rmse.append(metrics.mean_squared_error(y_test, prediction, squared=False))
                        mae.append(metrics.mean_absolute_error(y_test, prediction))
                    final_r2.append(np.mean(r2))
                    final_rmse.append(np.mean(rmse))
                    final_mae.append(np.mean(mae))
                    final_features.append(feature)
                fig, axs = plt.subplots(nrows=3)
                sns.barplot(x=final_features, y=final_r2, ax=axs[0])
                axs[0].set_ylabel('R2 Value')
                axs[0].set(xticklabels=[])
                sns.barplot(x=final_features, y=final_rmse, ax=axs[1])
                axs[1].set_ylabel('RMSE')
                axs[1].set(xticklabels=[])
                sns.barplot(x=final_features, y=final_mae, ax=axs[2])
                axs[2].set_ylabel('MAE')
                axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=40, ha="right")
                axs[0].set_title(str(folds) + '-Fold Stratified Cross-Validation with ' + str(repeats)
                                 + ' Repeats (' + category + ')')
                plt.savefig(path + category + '_Feature-Level.pdf', bbox_inches="tight")
                plt.close()
        else:
            feature_dfs = []
            feature_names = df.drop('CCP-score', axis=1).columns
            x = df.drop('CCP-score', axis=1).values
            r2, rmse, mae = [], [], []
            for train, test in cv.split(x, y):
                x_train, x_test = x[train], x[test]
                y_train, y_test = y[train], y[test]
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                r2.append(metrics.r2_score(y_test, prediction))
                rmse.append(metrics.mean_squared_error(y_test, prediction, squared=False))
                mae.append(metrics.mean_absolute_error(y_test, prediction))
                feature_dfs.append(pd.DataFrame(model.feature_importances_, index=feature_names))
            truths_df = pd.DataFrame(columns=['MeanR2', 'MeanRMSE', 'MeanMAE'])
            truths_df.iloc[0] = [np.mean(r2), np.mean(rmse), np.mean(mae)]
            truths_df.to_csv(path + 'Metrics.tsv', sep="\t")
            feature_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), feature_dfs)
            feature_df['mean'] = feature_df.mean(axis=1)
            feature_df = feature_df[['mean']].sort_values(by=['mean'], ascending=False)
            feature_df = feature_df.drop(feature_df[feature_df['mean'] == 0].index)
            feature_df.to_csv(path + 'FeatureImportances.tsv', sep="\t")
    elif model_type == 'XGBCategorical':  # regression build
        y = pd.factorize(df['Subtype'].values)[0]
        repeats = 100
        folds = int(min(Counter(y).values()))
        cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
        print("Conducting " + str(folds) + "-fold CV with " + str(repeats) + " repeats.")
        feature_names = df.drop('Subtype', axis=1).columns
        x = df.drop('Subtype', axis=1).values
        feature_dfs = []
        for train, test in cv.split(x, y):
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]
            model.fit(x_train, y_train)
            feature_dfs.append(pd.DataFrame(model.feature_importances_, index=feature_names))
        feature_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), feature_dfs)
        feature_df['mean'] = feature_df.mean(axis=1)
        feature_df = feature_df[['mean']].sort_values(by=['mean'], ascending=False)
        feature_df = feature_df.drop(feature_df[feature_df['mean'] == 0].index)
        feature_df.to_csv(path + 'FeatureImportances.tsv', sep="\t")
    elif test == 'bench':
        repeats = 10
        base_fpr = np.linspace(0, 1, 101)
        print("Conducting LOO with " + str(repeats) + " repeats.")
        bench_targets = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
        bench_colors = ['#1c9964', '#4b9634', '#768d00', '#a47d00', '#d35e00', '#ff0000']
        bench_palette = {bench_targets[i]: bench_colors[i] for i in range(len(bench_targets))}
        print(" . . . on " + str(len(bench_targets)) + " categories.")
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
        for category in bench_targets:
            sub_df = df.loc[df['TFX'] == category]
            y = pd.factorize(sub_df['Subtype'].values)[0]
            folds = int(min(Counter(y).values()))
            cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
            sub_df = sub_df.drop(['Subtype', 'TFX'], axis=1)
            x = sub_df.values
            if x.size == 0: continue
            tprs, aucs = [], []
            for train, test in cv.split(x, y):
                x_train, x_test = x[train], x[test]
                y_train, y_test = y[train], y[test]
                model.fit(x_train, y_train)
                prediction = model.predict_proba(x_test)
                fpr, tpr, _ = metrics.roc_curve(y_test, prediction[:, 1])
                auc = metrics.auc(fpr, tpr)
                aucs.append(auc)
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                tprs.append(tpr)
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            mean_auc = metrics.auc(base_fpr, mean_tprs)
            plt.plot(base_fpr, mean_tprs, color=bench_palette[category],
                     label='TFX = ' + str(category) + r': Mean ROC (AUC = % 0.2f )' % mean_auc, lw=2, alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(str(folds) + '-Fold Stratified Cross-Validation with ' + str(repeats) + ' Repeats (by tumor fraction)')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.savefig(path + 'TFX-Level_CV-ROC.pdf', bbox_inches="tight")
        plt.close()
    else:
        # extract sample info and truth values:
        y = pd.factorize(df['Subtype'].values)[0]
        samples = df.index.values
        # CV parameters (set to use minimum of 2 samples per subtype):
        repeats = 100
        folds = int(min(Counter(y).values()))  # LOO
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
        base_fpr = np.linspace(0, 1, 101)
        print("Conducting " + str(folds) + "-fold CV (Stratified) with " + str(repeats) + " repeats.")
        if group_level:
            df_bar = pd.DataFrame(columns=['Region', 'AUC', 'Feature'])
            categories = list(set([item.split('_')[1] for item in list(df.columns) if '_' in item]))
            if 'ATAC' in categories:
                categories = ['H3K4ME', 'H3K27AC', 'H3K27ME3', 'TSS', 'GB', 'TFBS-L', 'ATAC']
            else:
                categories = ['H3K4ME', 'H3K27AC', 'H3K27ME3', 'TSS', 'GB', 'TFBS-L']
            # features = list(set([item.split('_')[2] for item in list(df.columns) if '_' in item]))
            features = ['Central-Mean', 'Window-Mean', 'Jump-Amplitude', 'amplitude-ratio', 'peak-based-period',
                        'frag-mean', 'frag-cv', 'short-long-ratio']
            feature_colors = plt.cm.tab10(np.linspace(0, 1, len(features)))
            bar_palette = {features[i]: feature_colors[i] for i in range(len(features))}
            print(" . . . on " + str(len(categories)) + " categories.")
            for category in categories:
                plt.figure(figsize=(8, 8))
                for feature in features:
                    feature_name = category + '_' + feature
                    sub_df = df.filter(regex=feature_name)
                    x = sub_df.values
                    if x.size == 0: continue
                    tprs, aucs = [], []
                    for train, test in cv.split(x, y):
                        x_train, x_test = x[train], x[test]
                        y_train, y_test = y[train], y[test]
                        model.fit(x_train, y_train)
                        prediction = model.predict_proba(x_test)
                        fpr, tpr, _ = metrics.roc_curve(y_test, prediction[:, 1])
                        auc = metrics.auc(fpr, tpr)
                        aucs.append(auc)
                        tpr = np.interp(base_fpr, fpr, tpr)
                        tpr[0] = 0.0
                        tprs.append(tpr)
                        df_bar.loc[len(df_bar.index)] = [category, auc, feature]
                    tprs = np.array(tprs)
                    mean_tprs = tprs.mean(axis=0)
                    mean_auc = metrics.auc(base_fpr, mean_tprs)
                    plt.plot(base_fpr, mean_tprs, color=bar_palette[feature],
                             label=feature_name + r': Mean ROC (AUC = % 0.2f )' % mean_auc, lw=2, alpha=0.7)
                plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(str(folds) + '-Fold Stratified Cross-Validation with '
                          + str(repeats) + ' Repeats (' + category + ')')
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                plt.savefig(path + category + '_Feature-Level_CV-ROC.pdf', bbox_inches="tight")
                plt.close()
            plt.figure(figsize=(24, 8))
            sns.barplot(x='Region', y='AUC', hue='Feature', data=df_bar, palette=bar_palette)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(path + 'AUCBarPlot.pdf', bbox_inches="tight")
            plt.close()
            df_bar['Region_Feature'] = df_bar[['Region', 'Feature']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            df_print = df_bar[['Region_Feature', 'AUC']]
            df_print.groupby('Region_Feature').mean().reset_index().to_csv(path + 'AUCList.tsv', sep="\t")
        else:
            truths_df = pd.DataFrame(0, index=samples, columns=['Correct', 'Incorrect'])
            feature_dfs = []
            feature_names = df.drop('Subtype', axis=1).columns
            x = df.drop('Subtype', axis=1).values
            tprs, aucs = [], []
            plt.figure(figsize=(8, 8))
            for train, test in cv.split(x, y):
                x_train, x_test = x[train], x[test]
                y_train, y_test = y[train], y[test]
                model.fit(x_train, y_train)
                prediction = model.predict_proba(x_test)
                class_predictions = model.predict(x_test)
                fpr, tpr, _ = metrics.roc_curve(y_test, prediction[:, 1])
                aucs.append(metrics.auc(fpr, tpr))
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                tprs.append(tpr)
                feature_dfs.append(pd.DataFrame(model.feature_importances_, index=feature_names))
                for true, pred, label in zip(y_test, class_predictions, samples[test]):
                    if true == pred:
                        truths_df.at[label, 'Correct'] += 1
                    else:
                        truths_df.at[label, 'Incorrect'] += 1
                # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = % 0.2f)' % (i, roc_auc))
            truths_df['FractionCorrect'] = truths_df['Correct']/(truths_df['Correct'] + truths_df['Incorrect'])
            truths_df.to_csv(path + 'SamplePredictions.tsv', sep="\t")
            feature_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), feature_dfs)
            feature_df['mean'] = feature_df.mean(axis=1)
            feature_df = feature_df[['mean']].sort_values(by=['mean'], ascending=False)
            feature_df = feature_df.drop(feature_df[feature_df['mean'] == 0].index)
            feature_df.to_csv(path + 'FeatureImportances.tsv', sep="\t")
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            mean_auc = metrics.auc(base_fpr, mean_tprs)
            std = tprs.std(axis=0)
            tprs_upper = np.minimum(mean_tprs + std, 1)
            tprs_lower = mean_tprs - std
            plt.plot(base_fpr, mean_tprs, color='blue', label=r'Mean ROC (AUC = % 0.2f )' % mean_auc, lw=2, alpha=1)
            plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.2)
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(str(folds) + '-Fold Stratified Cross-Validation with ' + str(repeats) + ' Repeats')
            plt.legend(loc="lower right")
            plt.savefig(path + 'CV-ROC.pdf')
            plt.close()


def final_eval(y_test, predicted, predicted_prob, name):
    # accuracy and AUC
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob)
    print("Accuracy (overall correct predictions):", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    # precision and recall
    recall = metrics.recall_score(y_test, predicted)
    precision = metrics.precision_score(y_test, predicted)
    print("Recall (all classes predicted correctly):", round(recall, 2))
    print("Precision (confidence when predicting a class):", round(precision, 2))
    print("Details:")
    print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))
    # confusion matrix
    classes = np.unique(y_test)
    _, ax = plt.subplots()
    cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.savefig(name + '/' + name + '_ConfusionMatrix.pdf')
    plt.close()
    # ROC and precision/recall plotting
    _, ax = plt.subplots(nrows=1, ncols=2)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_prob)
    roc_auc = metrics.auc(fpr, tpr)
    ax[0].plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % roc_auc)
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, predicted_prob)
    roc_auc = metrics.auc(recalls, precisions)
    ax[1].plot(recalls, precisions, color='darkorange', lw=3, label='area = %0.2f' % roc_auc)
    ax[1].plot([0, 1], [(cm[1, 0] + cm[1, 0]) / len(y_test), (cm[1, 0] + cm[1, 0]) / len(y_test)], linestyle='--',
               color='navy', lw=3)
    ax[1].set(xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="lower left")
    ax[1].grid(True)
    plt.savefig(name + '/' + name + '_ROC-PR.pdf')
    plt.close()


def run_experiment(df, comparison, direct, model_type, individual, sub_string='', test=''):
    # If scale features is applicable (not a decision tree/random forest or node-based)
    # if model_type != 'RandomForestClassifier' and feature_type != 'ANOVA':
    #     scaled_features = StandardScaler().fit_transform(df.iloc[:, 1:].values)
    #     sub_df = pd.DataFrame(scaled_features, index=df.index, columns=df.iloc[:, 1:].columns)
    #     sub_df = pd.concat([df.iloc[:, :1], sub_df], axis=1, join='inner')
    #     df = sub_df
    # set path and make an identifier
    identifier = comparison + '_' + model_type + '_' + sub_string
    print("Running experiment " + identifier)
    if len(df.columns) < 3:
        print('This dataframe is empty! Moving on.')
        return
    if not os.path.exists(direct + identifier): os.makedirs(direct + identifier)
    # split data for training/testing (NOT validation, testing should not be used):
    # if test_fraction > 0:
    #     df_train, df_test = train_test_split(df, test_size=test_fraction, stratify=df['Subtype'], random_state=42)
    #     print("train shape:", df_train.drop("Subtype", axis=1).shape, "| test shape:",
    #           df_test.drop("Subtype", axis=1).shape)
    # else:  # use all data for training, e.g. with LuCaPs
    df_train = df
    print("train shape:", df_train.iloc[:, 1:].shape, "| NO TEST DATA HELD OUT")
    # get and evaluate model
    model = get_model(model_type, sample_size=df_train.shape[0])
    evaluate_model(df_train, model, direct + identifier + '/', individual, model_type, test)
        # x_train = df_train.drop('Subtype', axis=1).values
    # print("Finished evaluating: moving on to final training . . .")
    # # prepare model for final training/testing
    # y_train = pd.factorize(df_train['Subtype'].values)[0]
    # # final train and test
    # model.fit(x_train, y_train)
    # # print(model)
    # # print(model.data)
    # # save model
    # pickle.dump(model, open(identifier + '/' + identifier + '_model.pkl', 'wb'))
    # print("Finished.")


def test_model(df, model_path, name):
    samples = df.index
    x = df.values
    model = pickle.load(open(model_path, 'rb'))
    predictions = model.predict_proba(x)[:, 1]
    class_predictions = model.predict(x)
    predictions_df = pd.DataFrame(list(zip(predictions, class_predictions)), index=samples,
                                  columns=['ProbARPC', 'ARPC'])
    predictions_df['ProbNEPC'] = 1 - predictions_df['ProbARPC']
    predictions_df.to_csv(name + '_Predictions.tsv', sep="\t")


def product_column(a, b):
    ab = []
    for item_a in a:
        for item_b in b:
            ab.append(item_a + '_' + item_b)
    return ab


def subset_data(df, sub_list):
    regions = list(set([item.split('_')[0] for item in list(df.columns) if '_' in item]))
    categories = list(set([item.split('_')[1] for item in list(df.columns) if '_' in item]))
    features = list(set([item.split('_')[2] for item in list(df.columns) if '_' in item]))
    sub_list += [region for region in regions if any(gene + '-' in region for gene in sub_list)]
    sub_list = list(set(sub_list))
    all_features = product_column(categories, features)
    sub_features = product_column(sub_list, all_features)
    sub_df = df[df.columns.intersection(sub_features)]
    sub_df = sub_df[sub_df.columns.drop(list(sub_df.filter(regex='ATAC')))]  # remove ATAC-Seq peaks which muddle AR
    # return pd.concat([df['CCP-score'], sub_df], axis=1, join='inner')
    return pd.concat([df[set(df.columns).intersection({'Subtype', 'TFX'})], sub_df], axis=1, join='inner')


def main():
    test_name = 'bench'  # LuCaP or bench
    model_type = 'XGBoost'  # 'XGBoost, etc.
    if test_name == 'LuCaP':
        results_direct = '/fh/fast/ha_g/user/rpatton/ML_testing/LuCaP/'
        if not os.path.exists(results_direct): os.makedirs(results_direct)
        comparisons = {'PC-Phenotype': ['ARPC', 'NEPC']}
        pickl = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Exploration/LuCaP_FM.pkl'
        # data is formatted in the "ExploreFM.py" pipeline
        print("Loading " + pickl)
        df = pd.read_pickle(pickl)
    elif test_name == 'bench':
        results_direct = '/fh/fast/ha_g/user/rpatton/ML_testing/LuCaP_benchmarking/'
        if not os.path.exists(results_direct): os.makedirs(results_direct)
        comparisons = {'PC-Phenotype': ['ARPC', 'NEPC']}
        pickl = '/fh/fast/ha_g/user/rpatton/LuCaP_bench/Exploration/LuCaP_25X.pkl'
        # data is formatted in the "ExploreFM.py" pipeline
        print("Loading " + pickl)
        df = pd.read_pickle(pickl)
    else:  # e.g. mix
        results_direct = '/fh/fast/ha_g/user/rpatton/ML_testing/LuCaP_Healthy/'
        if not os.path.exists(results_direct): os.makedirs(results_direct)
        comparisons = {'Phenotype': ['ARPC', 'NEPC', 'Healthy']}
        pickl_1 = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Exploration/LuCaP_FM.pkl'
        pickl_2 = '/fh/fast/ha_g/user/rpatton/HD_data/Exploration/Healthy_FM.pkl'
        print("Loading " + pickl_1)
        df_1 = pd.read_pickle(pickl_1)
        df_1 = df_1.rename(columns={'PC-Phenotype': 'Phenotype'})
        print("Loading " + pickl_2)
        df_2 = pd.read_pickle(pickl_2)
        df_2.insert(0, 'Phenotype', 'Healthy')
        df = pd.concat([df_1, df_2], join='inner')
        print('Finished.')
    df = df[df.columns.drop(list(df.filter(regex='shannon-entropy')))]
    df = df[df.columns.drop(list(df.filter(regex='mean-depth')))]
    df = df[df.columns.drop(list(df.filter(regex='TFBS-S')))]
    df = df[df.columns.drop(list(df.filter(regex='ADLoss')))]
    df = df[df.columns.drop(list(df.filter(regex='NELoss')))]
    df = df[df.columns.drop(list(df.filter(regex='NEGain')))]
    # drop unused ATAC features
    df = df[df.columns.drop(list(df.filter(regex='TF-HD')))]
    df = df[df.columns.drop(list(df.filter(regex='TF-NoHD')))]
    df = df[df.columns.drop(list(df.filter(regex='NEGain')))]
    df = df[df.columns.drop(list(df.filter(regex='NELoss')))]
    df = df[df.columns.drop(list(df.filter(regex='ADGain')))]
    df = df[df.columns.drop(list(df.filter(regex='ADLoss')))]
    ####################################################################################################################
    # establish sub data frames
    # pam50 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/PAM50.txt', header=None)[0].tolist()
    # pcs37 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/PCS37.txt', header=None)[0].tolist()
    # ccp31 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/CCP31.txt', header=None)[0].tolist()
    pheno46 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/Pheno46.txt', header=None)[0].tolist()
    tf404 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/TF404.txt', header=None)[0].tolist()
    ar10 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/AR10.txt', header=None)[0].tolist()
    # run experiments
    print("Running experiments . . .")
    for comparison, test_list in comparisons.items():
        if comparison == 'LB-Phenotype':
            df_sub = pd.concat([df.filter(regex=comparison), df.filter(regex='TFX'), df.filter(regex='_')], axis=1)
            df_sub = df_sub.rename(columns={comparison: 'Subtype'})
            df_sub = df_sub[df_sub['Subtype'] != 'Dual']
        elif comparison == 'LB-Phenotype-NoNEPC':
            df_sub = df[df['PC-Phenotype'] != 'NEPC']
            df_sub = pd.concat([df_sub.filter(regex='LB-Phenotype'), df_sub.filter(regex='TFX'), df_sub.filter(regex='_')], axis=1)
            df_sub = df_sub.rename(columns={'LB-Phenotype': 'Subtype'})
            df_sub = df_sub[df_sub['Subtype'] != 'Dual']
        elif comparison == 'PC-Phenotype':
            df_sub = pd.concat([df.filter(regex=comparison), df.filter(regex='TFX'), df.filter(regex='_')], axis=1)
            df_sub = df_sub.rename(columns={comparison: 'Subtype'})
            df_sub = df_sub[df_sub['Subtype'] != 'AMPC']
            df_sub = df_sub[df_sub['Subtype'] != 'ARlow']
        else:  # mix, 'Phenotype'
            df_sub = pd.concat([df.filter(regex=comparison), df.filter(regex='TFX'), df.filter(regex='_')], axis=1)
            df_sub = df_sub.rename(columns={comparison: 'Subtype'})
            df_sub = df_sub[df_sub['Subtype'] != 'AMPC']
            df_sub = df_sub[df_sub['Subtype'] != 'ARlow']
        ################################################################################################################
        # categorical tests
        if test_name != 'bench':
            run_experiment(df_sub, comparison, results_direct, model_type, True, 'Region-Wise_ALL')
            run_experiment(subset_data(df_sub, pheno46), comparison, results_direct, model_type, True, 'Region-Wise_Pheno46')
            run_experiment(subset_data(df_sub, tf404), comparison, results_direct, model_type, True, 'Region-Wise_TF404')
            run_experiment(subset_data(df_sub, ar10), comparison, results_direct, model_type, True, 'Region-Wise_AR10')
        ################################################################################################################
        # individual tests
        # run_experiment(df_sub, comparison, results_direct, model_type, False, 'ALL', test_name)
        # run_experiment(subset_data(df_sub, pheno46), comparison, results_direct, model_type, False, 'Pheno46', test_name)
        # run_experiment(subset_data(df_sub, tf404), comparison, results_direct, model_type, False, 'TF404', test_name)
        # run_experiment(subset_data(df_sub, ar10), comparison, results_direct, model_type, False, 'AR10', test_name)
        ################################################################################################################
        # specific, promising groupings
        # sub_groups = ['_ATAC', '_TFBS-L', '_H3K4ME_frag-mean', '_H3K4ME_amplitude-ratio', '_H3K4ME_short-long',
        #               '_GB_amplitude-ratio', '_H3K27ME3_amplitude-ratio', '_TSS', '_GB', '_TFBS-L_Central-Mean']
        sub_groups = ['_ATAC']
        for sub_group in sub_groups:
            df_sub_group = pd.concat([df_sub[set(df_sub.columns).intersection({'Subtype', 'TFX'})], df_sub.filter(regex=sub_group)], axis=1)
            run_experiment(df_sub_group, comparison, results_direct, model_type, False, 'ALL' + sub_group, test_name)
            run_experiment(subset_data(df_sub_group, pheno46), comparison, results_direct, model_type, False, 'Pheno46' + sub_group, test_name)
            run_experiment(subset_data(df_sub_group, tf404), comparison, results_direct, model_type, False, 'TF404' + sub_group, test_name)
            run_experiment(subset_data(df_sub_group, ar10), comparison, results_direct, model_type, False, 'AR10' + sub_group, test_name)
        ################################################################################################################
        # categorical
        # sub_groups = ['_ATAC']
        # for sub_group in sub_groups:
        #     df_sub_group = pd.concat([df_sub['Subtype'], df_sub.filter(regex=sub_group)], axis=1)
        #     run_experiment(df_sub_group, comparison, results_direct, 'XGBCategorical', False, 'ALL' + sub_group)
        ################################################################################################################
        # regression on CCP score
        # ccp_labels = pd.read_table('/fh/fast/ha_g/user/rpatton/references/LuCaP_CCPScore.tsv',
        #                            sep='\t', index_col=0, names=['CCP-score'])
        # df_ccp = pd.merge(ccp_labels, df, left_index=True, right_index=True)
        # df_ccp = df_ccp.drop('PC-Phenotype', axis=1)
        # df_ccp = df_ccp.drop('LB-Phenotype', axis=1)
        # # run_experiment(df_ccp, comparison, results_direct, 'XGBRegressor', True, 'Region-Wise_ALL_CCPScore')
        # run_experiment(subset_data(df_ccp, pam50), comparison, results_direct, 'XGBRegressor', True,
        #                'Region-Wise_PAM50_CCPScore')
        # run_experiment(subset_data(df_ccp, pcs37), comparison, results_direct, 'XGBRegressor', True,
        #                'Region-Wise_PCS37_CCPScore')
        # run_experiment(subset_data(df_ccp, pheno46), comparison, results_direct, 'XGBRegressor', True,
        #                'Region-Wise_Pheno46_CCPScore')
        # run_experiment(subset_data(df_ccp, tf404), comparison, results_direct, 'XGBRegressor', True,
        #                'Region-Wise_TF404_CCPScore')
        # run_experiment(subset_data(df_ccp, ccp31), comparison, results_direct, 'XGBRegressor', True,
        #                'Region-Wise_CCP31_CCPScore')


if __name__ == "__main__":
    main()
