#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.0, 09/13/2021

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, norm, kruskal, spearmanr
from scipy.optimize import minimize_scalar
import scikit_posthocs as sp
from statsmodels.stats.multitest import fdrcorrection
from sklearn import metrics

targets = ['Healthy', 'ARPC', 'Luminal', 'NEPC', 'Basal', 'Patient', 'Gray', 'Indeterminate', 'MIX']
colors = ['#009988', '#0077BB', '#33BBEE', '#CC3311', '#EE7733', '#EE3377', '#BBBBBB', '#FFAE42', '#9F009F']
palette = {targets[i]: colors[i] for i in range(len(targets))}
interest_genes = ['AR', 'ASCL1', 'FOXA1', 'HOXB13', 'NKX3-1', 'REST', 'PGR', 'SOX2', 'ONECUT2', 'MYOG', 'MYF5']
sns.set(font_scale=1.5)
sns.set_style('ticks')


def fraction_plots(ref_dict, full_df, name):
    features = list(ref_dict.keys())
    # labels = pd.read_table(name + '/' + name + '_beta-predictions.tsv', sep='\t', index_col=0)
    # full_df = pd.merge(labels, full_df, left_index=True, right_index=True)
    # normalize = Normalize(0, 1)
    # cmap = LinearSegmentedColormap.from_list('', ['#CC3311', '#9F009F', '#0077BB'])
    for feature in features:
        df = pd.concat([full_df['TFX'], full_df['Subtype'], full_df.filter(regex=feature)], axis=1)
        x_arpc, y_arpc = df.loc[df['Subtype'] == 'ARPC', 'TFX'].values, df.loc[df['Subtype'] == 'ARPC', feature].values
        r_val_arpc, p_val_arpc = spearmanr(x_arpc, y_arpc)
        m_arpc, b_arpc = np.polyfit(x_arpc, y_arpc, 1)
        x_nepc, y_nepc = df.loc[df['Subtype'] == 'NEPC', 'TFX'].values, df.loc[df['Subtype'] == 'NEPC', feature].values
        r_val_nepc, p_val_nepc = spearmanr(x_nepc, y_nepc)
        m_nepc, b_nepc = np.polyfit(x_nepc, y_nepc, 1)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='TFX', y=feature, hue='Subtype', data=df, alpha=0.8, palette=palette, s=300)
        plt.plot(x_arpc, m_arpc * x_arpc + b_arpc, lw=2, color=palette['ARPC'])
        plt.plot(x_nepc, m_nepc * x_nepc + b_nepc, lw=2, color=palette['NEPC'])
        # scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        # scalarmappaple.set_array(df.values)
        # plt.colorbar(scalarmappaple, )
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.title(feature + ' vs Tumor Fraction' +
                  '\n ARPC: Spearman = ' + "{:e}".format(r_val_arpc) + ', p-val = ' + "{:e}".format(p_val_arpc) +
                  '\n NEPC: Spearman = ' + "{:e}".format(r_val_nepc) + ', p-val = ' + "{:e}".format(p_val_nepc))
        plt.savefig(name + '/' + feature + '_vsTFX.pdf', bbox_inches="tight")
        plt.close()


def dist_plots(full_df, name):
    features = list(set([item for item in list(full_df.columns) if '_' in item]))
    for feature_label in features:
        # format df for seaborn
        subs_key = full_df['Subtype']
        df = full_df.filter(regex=feature_label).transpose().melt()
        df = pd.merge(subs_key, df, left_index=True, right_on='variable')
        # histogram:
        # plt.figure(figsize=(8, 8))
        # sns.histplot(x='value', hue='Subtype', data=df, palette=palette, element="step")
        # plt.xlabel(feature_label)
        # plt.ylabel('Counts')
        # plt.title(feature_label + ' Histogram', size=14)
        # plt.savefig(name + '/' + feature_label + '_Histogram.pdf', bbox_inches="tight")
        # plt.close()
        # density plot
        plt.figure(figsize=(8, 8))
        sns.kdeplot(x='value', hue='Subtype', data=df, palette=palette, fill=True, common_norm=False)
        plt.xlabel(feature_label)
        plt.ylabel('Density')
        plt.title(feature_label + ' Kernel Density Estimation', size=14)
        plt.savefig(name + '/' + feature_label + '_Density.pdf', bbox_inches="tight")
        plt.close()


def box_plots(df, name):
    df = df[df.columns.drop(list(df.filter(regex='Window')))]
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.melt(id_vars='Subtype', var_name='Feature', value_name='Value', ignore_index=False)
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Feature', y='Value', hue='Subtype', data=df, palette=palette)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel('Counts')
    plt.title(name + ' Feature Distributions', size=14)
    plt.savefig(name + '/' + name + '_BoxPlot.pdf', bbox_inches="tight")
    plt.close()


def dist_plots_sample(full_df, name):
    features = list(set([item for item in list(full_df.columns) if '_' in item]))
    for feature_label in features:
        # format df for seaborn
        subs_key = full_df['Subtype']
        df = full_df.filter(regex=feature_label).transpose().melt()
        df = pd.merge(subs_key, df, left_index=True, right_on='variable')
        print(df)
        # plot_range = [0, 2 * np.mean(df['value'])]
        for key in subs_key:
            # density plot
            plt.figure(figsize=(8, 8))
            sns.kdeplot(x='value', hue='variable', data=df[df['Subtype'] == key], fill=True, common_norm=False)
            plt.xlabel(feature_label)
            plt.ylabel('Density')
            plt.title(key + ' ' + name + ' Kernel Density Estimation', size=14)
            plt.savefig(name + '/' + name + '_' + feature_label + '_Sample-Wise_'
                        + key + '_Density.pdf', bbox_inches="tight")
            plt.close()


def diff_exp_tw(df, name, thresh=0.05, sub_name=''):
    print('Conducting three-way differential expression analysis . . .')
    types = list(df.Subtype.unique())
    df_t1 = df.loc[df['Subtype'] == types[0]].drop('Subtype', axis=1)
    df_t2 = df.loc[df['Subtype'] == types[1]].drop('Subtype', axis=1)
    df_t3 = df.loc[df['Subtype'] == types[2]].drop('Subtype', axis=1)
    df_lpq = pd.DataFrame(index=df_t1.transpose().index, columns=['p-value', 'DunnSigPairs'])
    for roi in list(df_t1.columns):
        x, y, z = df_t1[roi].values, df_t2[roi].values, df_t3[roi].values
        if np.count_nonzero(~np.isnan(x)) < 2 or np.count_nonzero(~np.isnan(y)) < 2 or np.count_nonzero(~np.isnan(z)) < 2:
            continue
        try:
            kw_score = kruskal(x, y, z, nan_policy='omit')[1]
        except ValueError:
            continue
        df_lpq.at[roi, 'p-value'] = kw_score
        if kw_score < thresh:
            pairs = 0
            dunn_scores = sp.posthoc_dunn([x, y, z])
            if dunn_scores[1][2] < thresh:
                pairs += 1
            if dunn_scores[1][3] < thresh:
                pairs += 1
            if dunn_scores[2][3] < thresh:
                pairs += 1
            df_lpq.at[roi, 'DunnSigPairs'] = pairs
        else:
            df_lpq.at[roi, 'DunnSigPairs'] = 0
    # now calculate p-adjusted (Benjamini-Hochberg corrected p-values)
    df_lpq = df_lpq.dropna(how='all')
    df_lpq['p-adjusted'] = fdrcorrection(df_lpq['p-value'])[1]
    df_lpq = df_lpq.infer_objects()
    df_lpq = df_lpq.sort_values(by=['p-adjusted'])
    df_lpq.to_csv(name + '/' + name + sub_name + '_three-way_rpq.tsv', sep="\t")
    features = list(df_lpq[(df_lpq['p-adjusted'] < thresh) & (df_lpq['DunnSigPairs'] == 3)].index)
    with open(name + '/' + name + sub_name + '_three-way_FeatureList.tsv', 'w') as f_output:
        for item in features:
            f_output.write(item + '\n')
    return pd.concat([df.iloc[:, :1], df.loc[:, df.columns.isin(features)]], axis=1, join='inner')


def diff_exp(df, name, thresh=0.05, sub_name=''):
    print('Conducting differential expression analysis . . .')
    types = list(df.Subtype.unique())
    df_t1 = df.loc[df['Subtype'] == types[0]].drop('Subtype', axis=1)
    df_t2 = df.loc[df['Subtype'] == types[1]].drop('Subtype', axis=1)
    df_lpq = pd.DataFrame(index=df_t1.transpose().index, columns=['ratio', 'p-value'])
    for roi in list(df_t1.columns):
        x, y = df_t1[roi].values, df_t2[roi].values
        if np.count_nonzero(~np.isnan(x)) < 2 or np.count_nonzero(~np.isnan(y)) < 2:
            continue
        df_lpq.at[roi, 'ratio'] = np.mean(x)/np.mean(y)
        df_lpq.at[roi, 'p-value'] = mannwhitneyu(x, y)[1]
    # now calculate p-adjusted (Benjamini-Hochberg corrected p-values)
    df_lpq['p-adjusted'] = fdrcorrection(df_lpq['p-value'])[1]
    df_lpq = df_lpq.sort_values(by=['p-adjusted'])
    df_lpq = df_lpq.infer_objects()
    df_lpq.to_csv(name + '/' + name + sub_name + '_rpq.tsv', sep="\t")
    features = list(df_lpq[(df_lpq['p-adjusted'] < thresh)].index)
    with open(name + '/' + name + sub_name + '_FeatureList.tsv', 'w') as f_output:
        for item in features:
            f_output.write(item + '\n')
    return pd.concat([df.iloc[:, :1], df.loc[:, df.columns.isin(features)]], axis=1, join='inner')


def metric_analysis(df, name):
    print('Calculating metric dictionary . . .')
    df = df.dropna(axis=1)
    features = list(df.iloc[:, 1:].columns)
    types = list(df.Subtype.unique())
    mat = {}
    for feature in features:
        sub_df = pd.concat([df.iloc[:, :1], df[[feature]]], axis=1, join='inner')
        mat[feature] = {'Feature': feature}
        for subtype in types:
            mat[feature][subtype + '_Mean'] = np.nanmean(
                sub_df[sub_df['Subtype'] == subtype].iloc[:, 1:].to_numpy().flatten())
            mat[feature][subtype + '_Std'] = np.nanstd(
                sub_df[sub_df['Subtype'] == subtype].iloc[:, 1:].to_numpy().flatten())
    pd.DataFrame(mat).to_csv(name + '/' + name + '_weights.tsv', sep="\t")
    return mat


def gaussian_mixture_model(ref_dict, df, subtypes, name):
    print('Running Gaussian Mixture Model Predictor on ' + name + ' . . . ')
    features = list(ref_dict.keys())
    samples = list(df.index)
    predictions = pd.DataFrame(0, index=df.index, columns=['LR', 'Prediction'])
    # latents = [0.5, 0.5]
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        score_mat = pd.DataFrame(1, index=features, columns=[subtypes[0], subtypes[1], 'LR'])
        for feature in features:
            try:
                feature_val = df.loc[sample, feature]
            except KeyError:
                continue
            exp_a = tfx * ref_dict[feature][subtypes[0] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_a = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[0] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            exp_b = tfx * ref_dict[feature][subtypes[1] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_b = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[1] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            range_a = [exp_a - 3 * std_a, exp_a + 3 * std_a]
            range_b = [exp_b - 3 * std_b, exp_b + 3 * std_b]
            range_min, range_max = [min([item for sublist in [range_a, range_b] for item in sublist]),
                                    max([item for sublist in [range_a, range_b] for item in sublist])]
            pdf_a = norm.pdf(feature_val, loc=exp_a, scale=std_a)
            pdf_b = norm.pdf(feature_val, loc=exp_b, scale=std_b)
            if np.isnan(pdf_a) or np.isnan(pdf_b) or pdf_a == 0 or pdf_b == 0\
                    or np.isinf(pdf_a) or np.isinf(pdf_b) or not range_min < feature_val < range_max:
                pdf_a = 1
                pdf_b = 1
            # score_mat.loc[feature, subtypes[0]] = pdf_a
            # score_mat.loc[feature, subtypes[1]] = pdf_b
            score_mat.loc[feature, 'LR'] = np.log(pdf_a / pdf_b)
            # plot features for specific samples
            # if sample in ['FH0200_E_2_A', 'FH0312_E_1_A', 'FH0486_E_2_A']:
            #     plt.figure(figsize=(8, 8))
            #     x = np.linspace(range_min, range_max, 100)
            #     plt.plot(x, norm.pdf(x, exp_a, std_a), c=colors[1], label='Shifted ARPC')
            #     plt.plot(x, norm.pdf(x, exp_b, std_b), c=colors[3], label='Shifted NEPC')
            #     plt.axvline(x=feature_val, c=colors[5], label='Sample Value')
            #     plt.ylabel('Density')
            #     plt.xlabel(feature)
            #     plt.legend()
            #     plt.title(sample + ' ' + feature + ' Shifted Curves and Sample Value', size=14)
            #     plt.savefig(name + '/' + name + '_' + sample + '_' + feature + '.pdf', bbox_inches="tight")
            #     plt.close()
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(score_mat)
        # gamma_a = score_mat[subtypes[0]].product(axis=0) * latents[0]
        # gamma_b = score_mat[subtypes[1]].product(axis=0) * latents[1]
        # marginal = gamma_a + gamma_b
        # print(str(gamma_a) + '\t' + str(gamma_b) + '\t' + str(marginal))
        # predictions.loc[sample, subtypes[0]] = gamma_a / marginal
        # predictions.loc[sample, subtypes[1]] = gamma_b / marginal
        # predictions.loc[sample, 'LR'] = np.log(gamma_a) - np.log(gamma_b)
        predictions.loc[sample, 'LR'] = score_mat['LR'].sum(axis=0)
        if predictions.loc[sample, 'LR'] > 2.3:
            predictions.loc[sample, 'Prediction'] = subtypes[0]
        elif predictions.loc[sample, 'LR'] < -2.3:
            predictions.loc[sample, 'Prediction'] = subtypes[1]
        else:
            predictions.loc[sample, 'Prediction'] = 'Indeterminate'
    predictions.to_csv(name + '/' + name + '_predictions.tsv', sep="\t")
    # print('Predictions:')
    # print(predictions)


def gaussian_mixture_model_v2(ref_dict, df, subtypes, name):
    print('Running Gaussian Mixture Model Predictor (non-bianry) on ' + name + ' . . . ')
    features = list(ref_dict.keys())
    samples = list(df.index)
    predictions = pd.DataFrame(0, index=df.index, columns=[subtypes[0], subtypes[1], 'Prediction'])
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        score_mat = pd.DataFrame(1, index=features, columns=[subtypes[0], subtypes[1]])
        for feature in features:
            try:
                feature_val = df.loc[sample, feature]
            except KeyError:
                continue
            exp_a = tfx * ref_dict[feature][subtypes[0] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_a = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[0] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            exp_b = tfx * ref_dict[feature][subtypes[1] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_b = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[1] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            range_a = [exp_a - 3 * std_a, exp_a + 3 * std_a]
            range_b = [exp_b - 3 * std_b, exp_b + 3 * std_b]
            range_min, range_max = [min([item for sublist in [range_a, range_b] for item in sublist]),
                                    max([item for sublist in [range_a, range_b] for item in sublist])]
            pdf_a = norm.pdf(feature_val, loc=exp_a, scale=std_a)
            pdf_b = norm.pdf(feature_val, loc=exp_b, scale=std_b)
            pdf_healthy = norm.pdf(feature_val, loc=ref_dict[feature]['Healthy_Mean'], scale=ref_dict[feature]['Healthy_Std'])
            if np.isnan(pdf_a) or np.isnan(pdf_healthy) or pdf_a == 0 or pdf_healthy == 0\
                    or np.isinf(pdf_a) or np.isinf(pdf_healthy) or not range_min < feature_val < range_max:
                score_mat.loc[feature, subtypes[0]] = 0
            else:
                score_mat.loc[feature, subtypes[0]] = np.log(pdf_a / pdf_healthy)
            if np.isnan(pdf_b) or np.isnan(pdf_healthy) or pdf_b == 0 or pdf_healthy == 0\
                    or np.isinf(pdf_b) or np.isinf(pdf_healthy) or not range_min < feature_val < range_max:
                score_mat.loc[feature, subtypes[1]] = 0
            else:
                score_mat.loc[feature, subtypes[1]] = np.log(pdf_b / pdf_healthy)
        predictions.loc[sample, subtypes[0]] = score_mat[subtypes[0]].sum(axis=0)
        predictions.loc[sample, subtypes[1]] = score_mat[subtypes[1]].sum(axis=0)
        ar_score = predictions.loc[sample, subtypes[0]]
        ne_score = predictions.loc[sample, subtypes[1]]
        if ar_score > 2.3 and ar_score > 2 * ne_score:
            predictions.loc[sample, 'Prediction'] = subtypes[0]
        elif ne_score > 2.3 and ne_score > 2 * ar_score:
            predictions.loc[sample, 'Prediction'] = subtypes[1]
        elif ar_score > 2.3 and ne_score > 2.3:
            predictions.loc[sample, 'Prediction'] = 'Amphicrine'
        else:
            predictions.loc[sample, 'Prediction'] = 'Indeterminate/DNPC'
    predictions.to_csv(name + '/' + name + '_categorical-predictions.tsv', sep="\t")


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def specificity_sensitivity(target, predicted, threshold):
    thresh_preds = np.zeros(len(predicted))
    thresh_preds[predicted > threshold] = 1
    cm = metrics.confusion_matrix(target, thresh_preds)
    return cm[1, 1] / (cm[1, 0] + cm[1, 1]), cm[0, 0] / (cm[0, 0] + cm[0, 1])


def nroc_curve(y_true, predicted, num_thresh=100):
    step = 1/num_thresh
    thresholds = np.arange(0, 1 + step, step)
    fprs, tprs = [], []
    for threshold in thresholds:
        y_pred = np.where(predicted >= threshold, 1, 0)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fprs.append(fp / (fp + tn))
        tprs.append(tp / (tp + fn))
    return fprs, tprs, thresholds


def beta_descent(ref_dict, df, subtypes, name, eval, order=None, base_df=None):
    print('Running Heterogeneous Beta Predictor on ' + name + ' . . . ')
    if not os.path.exists(name + '/'):
        os.makedirs(name + '/')
    features = list(ref_dict.keys())
    cols = subtypes
    cols.append('Prediction')
    samples = list(df.index)
    if eval == 'Bar':
        predictions = pd.DataFrame(0, index=df.index, columns=[subtypes[0], subtypes[1], 'TFX', 'Prediction', 'Depth',
                                                               subtypes[0] + '_PLL', subtypes[1] + '_PLL', 'JPLL'])
        feature_pdfs = pd.DataFrame(columns=['Sample', 'TFX', 'Feature', 'Value',
                                             subtypes[0] + '_s-mean', subtypes[1] + '_s-mean',
                                             subtypes[0] + '_s-std', subtypes[1] + '_s-std',
                                             subtypes[0] + '_pdf', subtypes[1] + '_pdf'])
    else:
        predictions = pd.DataFrame(0, index=df.index, columns=[subtypes[0], subtypes[1], 'TFX', 'Prediction',
                                                               subtypes[0] + '_PLL', subtypes[1] + '_PLL', 'JPLL'])
    predictions['Subtype'] = df['Subtype']
    i = 0
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        pdf_set_a, pdf_set_b = [], []
        if base_df is not None:  # recompute reference dictionary without samples
            sample_comp = sample.split('_')[0] + '_LuCaP'
            ref_dict = metric_analysis(base_df.drop(sample_comp), name)
        for feature in features:
            try:
                feature_val = df.loc[sample, feature]
            except KeyError:
                continue
            exp_a = tfx * ref_dict[feature][subtypes[0] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_a = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[0] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            exp_b = tfx * ref_dict[feature][subtypes[1] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_b = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[1] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            pdf_a = norm.pdf(feature_val, loc=exp_a, scale=std_a)
            pdf_b = norm.pdf(feature_val, loc=exp_b, scale=std_b)
            if np.isfinite(pdf_a) and np.isfinite(pdf_b) and pdf_a != 0 and pdf_b != 0:
                pdf_set_a.append(pdf_a)
                pdf_set_b.append(pdf_b)

            # feature_pdfs.loc[i] = [sample, tfx, feature, feature_val, exp_a, exp_b, std_a, std_b, pdf_a, pdf_b]
            i += 1

        def objective(theta):
            log_likelihood = 0
            for val_1, val_2 in zip(pdf_set_a, pdf_set_b):
                joint_pdf = theta * val_1 + (1 - theta) * val_2
                if joint_pdf > 0:
                    log_likelihood += np.log(joint_pdf)
            return -1 * log_likelihood

        def final_pdf(final_weight):
            log_likelihood_a, log_likelihood_b, jpdf = 0, 0, 0
            for val_1, val_2 in zip(pdf_set_a, pdf_set_b):
                joint_a, joint_b = final_weight * val_1, (1 - final_weight) * val_2
                joint_pdf = final_weight * val_1 + (1 - final_weight) * val_2
                if joint_a > 0:
                    log_likelihood_a += np.log(joint_a)
                if joint_b > 0:
                    log_likelihood_b += np.log(joint_b)
                if joint_pdf > 0:
                    jpdf += np.log(joint_pdf)
            return log_likelihood_a, log_likelihood_b, jpdf

        weight_1 = minimize_scalar(objective, bounds=(0, 1), method='bounded').x

        final_pdf_a, final_pdf_b, final_jpdf = final_pdf(weight_1)
        predictions.loc[sample, 'TFX'] = tfx
        if eval == 'Bar':
            predictions.loc[sample, 'Depth'] = df.loc[sample, 'Depth']
        predictions.loc[sample, 'JPLL'] = final_jpdf
        predictions.loc[sample, subtypes[0]], predictions.loc[sample, subtypes[1]] = np.round(weight_1, 4), np.round(1 - weight_1, 4)
        predictions.loc[sample, subtypes[0] + '_PLL'], predictions.loc[sample, subtypes[1] + '_PLL'] = final_pdf_a, final_pdf_b
        if predictions.loc[sample, subtypes[0]] > 0.9:
            predictions.loc[sample, 'Prediction'] = subtypes[0]
        elif predictions.loc[sample, subtypes[0]] < 0.1:
            predictions.loc[sample, 'Prediction'] = subtypes[1]
        elif predictions.loc[sample, subtypes[0]] > 0.5:
            predictions.loc[sample, 'Prediction'] = 'Mixed_' + subtypes[0]
        else:
            predictions.loc[sample, 'Prediction'] = 'Mixed_' + subtypes[1]
    predictions.to_csv(name + '/' + name + '_beta-predictions.tsv', sep="\t")
    # feature_pdfs.to_csv(name + '/' + name + '_feature-values_pdfs.tsv', sep="\t")

    # if eval == 'Bar':  # for benchmarking
    #     depths = ['0.2X', '1X', '25X']
    #     bench_targets = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    #     # bench_colors = ['#1c9964', '#4b9634', '#768d00', '#a47d00', '#d35e00', '#ff0000']
    #     # bench_palette = {bench_targets[i]: bench_colors[i] for i in range(len(bench_targets))}
    #     df_bar = pd.DataFrame(columns=['Depth', 'TFX', 'AUC'])
    #     for depth in depths:
    #         for category in bench_targets:
    #             sub_df = predictions.loc[predictions['TFX'] == category]
    #             sub_df = sub_df.loc[sub_df['Depth'] == depth]
    #             y = pd.factorize(sub_df['Subtype'].values)[0]
    #             fpr, tpr, threshold = metrics.roc_curve(y, sub_df['NEPC'].values)
    #             roc_auc = metrics.auc(fpr, tpr)
    #             df_bar.loc[len(df_bar.index)] = [depth, category, roc_auc]
    #     plt.figure(figsize=(12, 8))
    #     sns.barplot(x='TFX', y='AUC', hue='Depth', data=df_bar)
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #     plt.savefig(name + '/' + 'AUCBarPlot.pdf', bbox_inches="tight")
    #     plt.close()
    #     df_bar.to_csv(name + '/' + 'AUCList.tsv', sep="\t")

    if eval == 'Bar':  # for benchmarking
        depths = ['0.2X', '1X', '25X']
        bench_targets = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
        predictions = predictions[predictions['TFX'] != 0.03]
        for depth in depths:
            df = predictions.loc[predictions['Depth'] == depth]
            plt.figure(figsize=(8, 8))
            # sns.boxplot(x='TFX', y='NEPC', hue='Subtype', data=df, order=bench_targets, boxprops=dict(alpha=.3), palette=palette)
            sns.swarmplot(x='TFX', y='NEPC', hue='Subtype', palette=palette, data=df, s=10, alpha=0.8, dodge=False)
            plt.ylabel('NEPC Score')
            plt.xlabel('Tumor Fraction')
            plt.title('Benchmarking Scores at ' + depth, size=14, y=1.1)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.savefig(name + '/' + depth + '_BoxPlot.pdf', bbox_inches="tight")
            plt.close()

    if eval == 'SampleBar':  # for benchmarking
        import matplotlib.cm as cm
        from matplotlib.colors import LinearSegmentedColormap
        if order is not None:
            predictions = predictions.reindex(order.index)
        predictions = predictions.sort_values('NEPC')
        predictions['NEPC'] = predictions['NEPC'] - 0.3314
        data = predictions.groupby(predictions['NEPC']).size()
        cmap = LinearSegmentedColormap.from_list('', ['#0077BB', '#CC3311'])
        cm.register_cmap("mycolormap", cmap)
        if order is not None:
            predictions = predictions.reindex(order.index)
        pal = sns.color_palette("mycolormap", len(data))
        sns.set_context(rc={'patch.linewidth': 0.0})
        plt.figure(figsize=(32, 2))
        g = sns.barplot(x=predictions.index, y='NEPC', hue='NEPC', data=predictions, palette=pal, dodge=False)
        g.legend_.remove()
        sns.scatterplot(x=predictions.index, y='NEPC', hue='NEPC', data=predictions, palette=pal, s=600, legend=False)

        def change_width(ax, new_value):
            for patch in ax.patches:
                current_width = patch.get_width()
                diff = current_width - new_value
                # we change the bar width
                patch.set_width(new_value)
                # we recenter the bar
                patch.set_x(patch.get_x() + diff * .5)

        change_width(g, .2)
        for item in g.get_xticklabels():
            item.set_rotation(45)
        plt.axhline(y=0, color='b', linestyle='--', lw=2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(name + '/PredictionBarPlot.pdf', bbox_inches="tight")
        plt.close()

    if eval == 'AUCBar':  # for benchmarking
        df_bar = pd.DataFrame(columns=['TFX', 'Depth', 'AUC'])
        bench_targets = [0.01, 0.05, 0.1, 0.2, 0.3]
        for category in bench_targets:
            for depth in ['0.2X', '1X', '25X']:
                sub_df = predictions[(df['TFX'] == category) & (df['Depth'] == depth)]
                y = pd.factorize(sub_df['Subtype'].values)[0]
                fpr, tpr, _ = metrics.roc_curve(y, sub_df['NEPC'])
                auc = metrics.auc(fpr, tpr)
                df_bar.loc[len(df_bar.index) + 1] = [category, depth, auc]
        plt.figure(figsize=(8, 8))
        sns.barplot(x='TFX', y='AUC', hue='Depth', data=df_bar)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(name + '/AUCBarPlot.pdf', bbox_inches="tight")
        plt.close()
        df_bar.to_csv(name + '/AUCList.tsv', sep="\t")

    if eval == 'ROC':
        predictions = predictions[predictions['Subtype'].isin(['ARPC', 'NEPC'])]
        thresholds = pd.DataFrame(0, index=['AllTFX', '0.00-0.10', '0.10-1.00'],
                                  columns=['OptimumThreshold', 'Sensitivity', 'Specificity'])
        # All TFXs
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
        y = pd.factorize(predictions['Subtype'].values)[0]
        fpr, tpr, threshold = metrics.roc_curve(y, predictions['NEPC'].values)
        # fpr, tpr, threshold = nroc_curve(y, predictions['NEPC'].values)
        pd.DataFrame([threshold, tpr, [1 - val for val in fpr]],
                     index=['Threshold', 'Sensitivity', 'Specificity'],
                     dtype=float).transpose().to_csv(name + '/' + name + '_AllThresholds.tsv', sep="\t")
        roc_auc = metrics.auc(fpr, tpr)
        optimum_thresh = Find_Optimal_Cutoff(y, predictions['NEPC'].values)[0]
        specificity, sensitivity = specificity_sensitivity(y, predictions['NEPC'].values, optimum_thresh)
        print(specificity_sensitivity(y, predictions['NEPC'].values, 0.3314))
        thresholds.loc['AllTFX'] = [optimum_thresh, specificity, sensitivity]
        plt.plot(fpr, tpr, label='AUC = % 0.2f' % roc_auc, lw=4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.savefig(name + '/' + name + '_ROC.pdf', bbox_inches="tight")
        plt.close()
        # by TFX
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
        # 0.00 - 0.10
        sub_df = predictions.loc[predictions['TFX'] < 0.10]
        y = pd.factorize(sub_df['Subtype'].values)[0]
        fpr, tpr, threshold = metrics.roc_curve(y, sub_df['NEPC'].values, drop_intermediate=False)
        # fpr, tpr, threshold = nroc_curve(y, sub_df['NEPC'].values)
        pd.DataFrame([threshold, tpr, [1 - val for val in fpr]],
                     index=['Threshold', 'Sensitivity', 'Specificity'],
                     dtype=float).transpose().to_csv(name + '/' + name + '_0.00-0.10Thresholds.tsv', sep="\t")
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label='TFX < 0.10: AUC = % 0.2f' % roc_auc, lw=4, color='#1c9964')
        optimum_thresh = Find_Optimal_Cutoff(y, sub_df['NEPC'].values)[0]
        specificity, sensitivity = specificity_sensitivity(y, sub_df['NEPC'].values, optimum_thresh)
        thresholds.loc['0.00-0.10'] = [optimum_thresh, specificity, sensitivity]
        # 0.25 - 1.00
        sub_df = predictions.loc[predictions['TFX'] > 0.25]
        y = pd.factorize(sub_df['Subtype'].values)[0]
        fpr, tpr, threshold = metrics.roc_curve(y, sub_df['NEPC'].values)
        # fpr, tpr, threshold = nroc_curve(y, sub_df['NEPC'].values)
        pd.DataFrame([threshold, tpr, [1 - val for val in fpr]],
                     index=['Threshold', 'Sensitivity', 'Specificity'],
                     dtype=float).transpose().to_csv(name + '/' + name + '_0.1-1.00Thresholds.tsv', sep="\t")
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label='TFX > 0.10: AUC = % 0.2f' % roc_auc, lw=4, color='#ff0000')
        optimum_thresh = Find_Optimal_Cutoff(y, sub_df['NEPC'].values)[0]
        specificity, sensitivity = specificity_sensitivity(y, sub_df['NEPC'].values, optimum_thresh)
        thresholds.loc['0.10-1.00'] = [optimum_thresh, specificity, sensitivity]
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.savefig(name + '/' + name + '_TFX-ROC.pdf', bbox_inches="tight")
        plt.close()
        thresholds.to_csv(name + '/' + name + '_Thresholds.tsv', sep="\t")


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
    return pd.concat([df['Subtype'], sub_df], axis=1, join='inner')


def main():
    test_data = 'bench'  # bench or patient_ULP/WGS or freed
    # LuCaP dataframe - data is formatted in the "ExploreFM.py" pipeline
    pickl = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Exploration/LuCaP_FM.pkl'
    print("Loading " + pickl)
    df = pd.read_pickle(pickl)
    df = df.drop('LB-Phenotype', axis=1)
    df = df.rename(columns={'PC-Phenotype': 'Subtype'})
    df = df[df['Subtype'] != 'AMPC']
    df = df[df['Subtype'] != 'ARlow']
    df = df[df.columns.drop(list(df.filter(regex='shannon-entropy')))]
    df_lucap = df[df.columns.drop(list(df.filter(regex='mean-depth')))]
    # Healthy dataframe - data is formatted in the "ExploreFM.py" pipeline
    pickl = '/fh/fast/ha_g/user/rpatton/HD_data/Exploration/Healthy_FM.pkl'
    print("Loading " + pickl)
    df = pd.read_pickle(pickl)
    df.insert(0, 'Subtype', 'Healthy')
    df = df[df.columns.drop(list(df.filter(regex='shannon-entropy')))]
    df_hd = df[df.columns.drop(list(df.filter(regex='mean-depth')))]
    # Patient dataframe - data is formatted in the "ExploreFM.py" pipeline
    if test_data == 'patient_WGS':
        labels = pd.read_table('/fh/fast/ha_g/user/rpatton/patient-WGS_data/WGS_TF_hg19.txt',
                               sep='\t', index_col=0, names=['TFX'])
        pickl = '/fh/fast/ha_g/user/rpatton/patient-WGS_data/Exploration/Patient_FM.pkl'
        print("Loading " + pickl)
        df = pd.read_pickle(pickl)
        df = pd.merge(labels, df, left_index=True, right_index=True)
        df['Subtype'] = 'ARPC'
        # truths = pd.read_table('/fh/fast/ha_g/user/rpatton/references/patient_subtypes.tsv',
        #                        sep='\t', index_col=0, names=['Subtype'])
        # df = pd.merge(truths, df, left_index=True, right_index=True)
        df = df[df.columns.drop(list(df.filter(regex='shannon-entropy')))]
        df_patient = df[df.columns.drop(list(df.filter(regex='mean-depth')))]
        ordering = pd.read_table('/fh/fast/ha_g/user/rpatton/ML_testing/Generative/Samples_WGS.txt',
                                 sep='\t', index_col=0, header=None)
    elif test_data == 'patient_ULP':
        labels = pd.read_table('/fh/fast/ha_g/user/rpatton/patient-ULP_data/ULP_TF_hg19.txt',
                               sep='\t', index_col=0, names=['TFX'])
        pickl = '/fh/fast/ha_g/user/rpatton/patient-ULP_data/Exploration/Patient_FM.pkl'
        print("Loading " + pickl)
        df = pd.read_pickle(pickl)
        df = pd.merge(labels, df, left_index=True, right_index=True)
        df['Subtype'] = 'ARPC'
        # truths = pd.read_table('/fh/fast/ha_g/user/rpatton/references/patient_subtypes.tsv',
        #                        sep='\t', index_col=0, names=['Subtype'])
        # df = pd.merge(truths, df, left_index=True, right_index=True)
        df = df[df.columns.drop(list(df.filter(regex='shannon-entropy')))]
        df_patient = df[df.columns.drop(list(df.filter(regex='mean-depth')))]
        ordering = pd.read_table('/fh/fast/ha_g/user/rpatton/ML_testing/Generative/Samples_ULP.txt',
                                 sep='\t', index_col=0, header=None)
    elif test_data == 'patient_freed':
        pickl = '/fh/fast/ha_g/user/rpatton/Freedman_data/Exploration/Freedman_FM.pkl'
        print("Loading " + pickl)
        df_patient = pd.read_pickle(pickl)
    else:  # bench
        print("Loading benchmarking pickles")
        df_1 = pd.read_pickle('/fh/fast/ha_g/user/rpatton/LuCaP_bench/Exploration/LuCaP_25X.pkl')
        df_2 = pd.read_pickle('/fh/fast/ha_g/user/rpatton/LuCaP_bench/Exploration/LuCaP_1X.pkl')
        df_3 = pd.read_pickle('/fh/fast/ha_g/user/rpatton/LuCaP_bench/Exploration/LuCaP_0.2X.pkl')
        df_1.insert(0, 'Depth', '25X')
        df_2.insert(0, 'Depth', '1X')
        df_3.insert(0, 'Depth', '0.2X')
        df_patient = pd.concat([df_1, df_2, df_3])
        df_patient = df_patient.rename(columns={'PC-Phenotype': 'Subtype'})
        df_patient = df_patient[~df_patient.index.str.contains('NPH014')]
        df_patient = df_patient[~df_patient.index.str.contains('136_')]
        df_patient = df_patient[~df_patient.index.str.contains('145-1_')]
    ####################################################################################################################
    df_train = pd.concat([df_lucap, df_hd])
    df_train = df_train[df_train.columns.drop(list(df_train.filter(regex='TFBS-S')))]
    df_train = df_train[df_train.columns.drop(list(df_train.filter(regex='ADLoss')))]
    df_train = df_train[df_train.columns.drop(list(df_train.filter(regex='NELoss')))]
    df_train = df_train[df_train.columns.drop(list(df_train.filter(regex='NEGain')))]
    df_train = df_train[df_train.columns.drop(list(df_train.filter(regex='Jump-Amplitude')))]
    ####################################################################################################################
    # establish sub data frames
    # pam50 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/PAM50.txt', header=None)[0].tolist()
    # pcs37 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/PCS37.txt', header=None)[0].tolist()
    # pheno46 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/Pheno46.txt', header=None)[0].tolist()
    # tf404 = pd.read_table('/fh/fast/ha_g/user/rpatton/references/TF404.txt', header=None)[0].tolist()
    # run experiments
    print("Running experiments . . .")
    # features = ['ATAC-TFBS', 'ATAC-TF', 'ATAC-TF-HD', 'TFBS', 'GB_amplitude-ratio', 'ATAC-amp']
    features = ['ATAC-TF']
    for feature in features:
        name = 'ARPC-NEPC_' + feature
        if not os.path.exists(name + '/'): os.makedirs(name + '/')
        if feature == 'ATAC-TFBS':
            df_sub_atac = pd.concat([df_train['Subtype'], df_train.filter(regex='_ATAC')], axis=1)
            df_diff_atac = diff_exp(df_sub_atac, name, sub_name='-ATAC')
            df_sub_tfbs = pd.concat([df_train['Subtype'], df_train.filter(regex='_TFBS')], axis=1)
            df_diff_tfbs = diff_exp(df_sub_tfbs, name, sub_name='-TFBS')
            df_diff = df_diff_atac.merge(df_diff_tfbs.iloc[:, 1:], left_index=True, right_index=True)
        elif feature == 'ATAC-amp':
            df_sub_atac = pd.concat([df_train['Subtype'], df_train.filter(regex='_ATAC')], axis=1)
            df_diff_atac = diff_exp(df_sub_atac, name, sub_name='-ATAC')
            df_sub_tfbs = pd.concat([df_train['Subtype'], df_train.filter(regex='_TFBS')], axis=1)
            df_diff_tfbs = diff_exp(df_sub_tfbs, name, sub_name='GB_amplitude-ratio')
            df_diff = df_diff_atac.merge(df_diff_tfbs.iloc[:, 1:], left_index=True, right_index=True)
        elif feature == 'ATAC-TF':
            df_sub = pd.concat([df_train['Subtype'], df_train.filter(regex=r'ATAC-.*-TF_')], axis=1)
            df_diff = diff_exp(df_sub, name)
        elif feature == 'ATAC-TF-HD':
            df_sub = pd.concat([df_train['Subtype'], df_train.filter(regex=r'ATAC-.*HD')], axis=1)
            df_diff = diff_exp(df_sub, name)
        else:  # (TFBS, GB_amplitude-ratio)
            df_sub = pd.concat([df_train['Subtype'], df_train.filter(regex='_' + feature)], axis=1)
            df_diff = diff_exp(df_sub, name)
        # dist_plots(df_diff, name)
        # box_plots(df_diff, name)
        # dist_plots_sample(df_diff, test_type, 'p05_' + feature)
        # dist_plots_sample(df_patient, test_type, 'p05_' + feature)
        # for sample in list(df_patient.index):
        #     dist_plots(pd.concat([df_diff, df_patient.loc[[sample]]], join='inner'), test_type,
        #                'p05_' + feature + '_' + sample)
        # gaussian_mixture_model(metric_dict, df_patient, ['ARPC', 'NEPC'], name)
        # gaussian_mixture_model_v2(metric_dict, df_patient, ['ARPC', 'NEPC'], name)
        metric_dict = metric_analysis(df_diff, name)
        if 'patient' in test_data:
            beta_descent(metric_dict, df_patient, ['ARPC', 'NEPC'], name, eval='SampleBar', order=ordering)
            # fraction_plots(metric_dict, df_patient, name)
        else:  # benchmarking
            beta_descent(metric_dict, df_patient, ['ARPC', 'NEPC'], name, eval='AUCBar', base_df=df_diff)


if __name__ == "__main__":
    main()
