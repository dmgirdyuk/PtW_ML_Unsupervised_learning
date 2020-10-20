import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def plot_ad(df, models, outliers_fraction):
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Sales', 'Profit']] = scaler.fit_transform(df[['Sales', 'Profit']])
    X1 = df['Sales'].values.reshape(-1, 1)
    X2 = df['Profit'].values.reshape(-1, 1)
    
    X = np.concatenate((X1, X2), axis=1)
    fig = plt.figure(figsize=(30, 30))
    
    for i, (clf_name, clf) in enumerate(models.items()):
        clf.fit(X)
        scores_pred = clf.decision_function(X) * - 1
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)

        df1 = df
        df1['outlier'] = y_pred.tolist()
            
        # sales - inlier feature 1,  profit - inlier feature 2
        inliers_sales = np.array(df1['Sales'][df1['outlier'] == 0]).reshape(-1, 1)
        inliers_profit = np.array(df1['Profit'][df1['outlier'] == 0]).reshape(-1, 1)
            
        # sales - outlier feature 1, profit - outlier feature 2
        outliers_sales = df1['Sales'][df1['outlier'] == 1].values.reshape(-1, 1)
        outliers_profit = df1['Profit'][df1['outlier'] == 1].values.reshape(-1, 1)
                 
        threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)     
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)
        
        subplot = plt.subplot(3, 3, i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
        b = subplot.scatter(inliers_sales, inliers_profit, c='white', s=20, edgecolor='k')
        c = subplot.scatter(outliers_sales, outliers_profit, c='black', s=20, edgecolor='k')
        subplot.axis('tight')
        
        subplot.legend([a.collections[0], b, c],
                       ['learned decision function', f'inliers {n_inliers}', f'outliers {n_outliers}'],
                       prop=matplotlib.font_manager.FontProperties(size=15), loc='lower right')
        subplot.set_xlim((0, 1))
        subplot.set_ylim((0, 1))
        subplot.set_xlabel(f'{i + 1} {clf_name}', fontsize=15)

    plt.close(fig)
    return fig
