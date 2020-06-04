import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_historyplt(
                    history:dict,
                    xlabel='Epochs',
                    ylabel='Loss',
                    title='Model training history'
                ) -> None:
    """Plot history of model training during epochs as a function of chosen
    loss paramter (e.g. mean absolute error, MAE). The plot will shop the
    error from both training and testing of the model."""

    for key, value in history.items():
        plt.plot(value, label=key)
        _add_plt_properties(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.show()

def get_predplt(
                real:pd.DataFrame,
                pred:pd.DataFrame,
                signal:str,
                unit:str,
                label_real='actual',
                label_pred='predicted',
                xlabel=''
            ) -> None:
    """Plot actual and predicted values."""

    plt.plot(real, color='red', label=label_real)
    plt.plot(pred, color='blue',label=label_pred)
    _add_plt_properties(xlabel='', ylabel=unit, title=signal)
    plt.show()
    return

def get_distplt(
                    signal:str,
                    dist:pd.DataFrame,
                    threshold:int,
                    unit:str=None
            ) -> None:
    """Plot distributions based on absolute errors and threshold values."""
    if unit is not None: label = f'Absolute error ({unit})'
    else: label = 'Absolute_error'
    below_t = dist[dist < threshold]
    above_t = dist[dist > threshold]

    f, axes = plt.subplots(2,2)
    plt.title(
        "Distribution plots for the absolute prediction error for "\
            f"'{signal}' with threshold: {threshold:.2f}"
        )
    # Plot 1 - complete distribution -----------------------------------------
    sns.distplot(
            dist,kde=True,kde_kws={'label':'KDE','color':'black'},ax=axes[0,0]
        )
    axes[0,0].axvline(
                        threshold, label=f'Threshold: {threshold:.2f}',
                        linestyle='dashed', linewidth=2, color='red'
                    )
    axes[0,0].set_xlabel(label)
    axes[0,0].legend()
    axes[0,0].set_title(f'Complete distribution (n: {dist.shape[0]})')

    # Plot 2 - combined distribution -----------------------------------------
    sns.distplot(below_t, kde=True, color='green', ax=axes[0,1],
                    kde_kws={'label':f'Above (n: {below_t.shape[0]})'})
    sns.distplot(above_t, kde=True, color='red', ax=axes[0,1],
                    kde_kws={'label':f'Above (n: {above_t.shape[0]})'})
    axes[0,1].axvline(
                        threshold, label=f'Threshold: {threshold:.2f}',
                        linestyle='dashed', linewidth=2, color='red'
                    )
    axes[0,1].set_xlabel(label)
    axes[0,1].legend()
    axes[0,1].set_title(f'Combined distributions')

    # Plot 3 - distribution below threshold ----------------------------------
    sns.distplot(
                    below_t,kde=True,color='green',
                    kde_kws={'label':'KDE','color':'black'},ax=axes[1,0]
    )
    axes[1,0].set_xlabel(label)
    axes[1,0].legend()
    axes[1,0].set_title(f'Below threshold (n: {below_t.shape[0]})')

    # Plot 4 - distribution above threshold ----------------------------------
    sns.distplot(
                    above_t,kde=True,color='red',
                    kde_kws={'label':'KDE','color':'black'},ax=axes[1,1]
            )
    axes[1,1].set_xlabel(label)
    axes[1,1].legend()
    axes[1,1].set_title(f'Above threshold (n: {above_t.shape[0]})')

    # Render plots -----------------------------------------------------------
    plt.tight_layout()
    plt.show()
    return

def loss_threshold(
                        signal:str,
                        loss:pd.DataFrame,
                        threshold:int,
                        unit:str=None,
                        y_label='Absolute error',
                        thresh_label='anomaly threshold'
                ) -> None:
    """Plot the loss of each timestep against the threshold values, indicating
    where anomalies are located in the dataset."""
    if unit is not None: ylabel = f'{y_label} ({unit})'
    title = f"Loss values of '{signal}'"
    above = copy.deepcopy(loss)
    below = copy.deepcopy(loss)
    above[above < threshold] = np.nan
    below[below >= threshold] = np.nan
    plt.plot(below, color='blue', label=f'below (n:{below.count()})')
    plt.plot(above, color='red', label=f'above (n:{above.count()})')
    plt.axhline(
                threshold,
                color='black',
                linestyle='dashed',
                label=f'threshold: {threshold:.2f}'
            )
    _add_plt_properties(xlabel='',ylabel=ylabel,title=title)
    plt.show()
    return

def get_anomalyplt(
                    signal:str,
                    real:pd.DataFrame,
                    anoms:pd.DataFrame,
                    unit:str=None,
                ) -> None:
    """Plot the actual state values together with registered anomalies. The
    anomalies are plotted as single points corresponding with the state value
    at the given anomaly timestep. """
    anom_dps = real[anoms==True]
    plt.scatter(anom_dps.index, anom_dps, color='red', label=f'anomaly (n:{anom_dps.count()})')
    plt.plot(real, label=f'state values (n:{real.count()})')
    plt.legend()
    plt.show()
    return

def _add_plt_properties(
                        xlabel:str,
                        ylabel:str,
                        title:str,
                        x_fontsize:int=14,
                        y_fontsize:int=14,
                        t_fontsize:int=20
                    ) -> None:
    plt.xlabel(xlabel, fontsize=x_fontsize)
    plt.ylabel(ylabel, fontsize=y_fontsize)
    plt.title(title, fontsize=t_fontsize)
    plt.legend(fontsize=14)
    plt.get_current_fig_manager().window.state('zoomed')
    return

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
