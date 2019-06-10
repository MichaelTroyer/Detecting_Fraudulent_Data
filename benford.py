# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:48:10 2019

@author: mtroyer

Detecting fraduelent data using Benford's Law and Pearsons's Chi-Squared test.

https://en.wikipedia.org/wiki/Benford%27s_law

Benford's law states that in many naturally occurring collections of numbers,
the leading significant digit is likely to be small. For example, in sets
that obey the law, the number 1 appears as the leading significant digit
about 30% of the time, while 9 appears as the leading significant digit
less than 5% of the time. If the digits were distributed uniformly, they
would each occur about 11.1% of the time.

Based on Impractical Python: Chapter 16 Finding Frauds with Benford's Law
by Lee Vaughan

Sample data from:
https://github.com/rlvaugh/Impractical_Python_Projects/tree/master/Chapter_16

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import stats


BENFORD_PROPS = [.301, .176, .125, .097, .079, .067, .058, .051, .046]


def load_data(data_file, header=False):
    """Read an input data file and return a pandas series object."""
    _, ext = os.path.splitext(data_file)
    
    if ext.lower() in ('.csv', '.txt'):
        if header:
            df = pd.read_csv(data_file)
            return df[df.columns[0]]
        else:
            df = pd.read_csv(data_file, header=None)
            return df[0]
    if ext.lower() in ('.xls', '.xlsx'):
        if header:
            df = pd.read_excel(data_file)
            return df[df.columns[0]]
        else:
            df = pd.read_excel(data_file, header=None)
            return df[0]
    else:
        raise IOError('Cannot process file format: {}'.format(ext))
    

def count_first_digits(pandas_series):
    """Count first digits in a pandas series, return digits and observed frequency"""
    first_digits = {i: 0 for i in range(1, 10)}  # Make sure all digits are represented
    for value in pandas_series:
        if isinstance(value, (int, float)):
            first_digit = int(str(value)[0])
        elif isinstance(value, str):
            first_digit = int(value[0])
        else:
            raise ValueError('Wrong data format: {}'.format(type(value)))
        
        first_digits[first_digit] += 1
    return [cnt for _, cnt in sorted(first_digits.items())]


def get_expected_counts(n_observations):
    """Use Benford's proportions to determine expected counts."""
    return [round(n_observations * prop) for prop in BENFORD_PROPS]


def chi_squared_test(actual, expected):
    chi2_stat, p = stats.chisquare(actual, expected)
    return chi2_stat, p


def plot_results(actuals):
    """Make bar chart of observed vs expected 1st digit proportion."""
    sum_actuals = float(sum(actuals))
    proportions = [actual / sum_actuals for actual in actuals]
    
    fig, ax = plt.subplots()

    index = [i + 1 for i in range(len(proportions))]

    ax.set_title('First Digit Proportion vs. Benford Proportions', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels(index, fontsize=12)
    ax.set_xlabel('First Digit', fontsize=12)
 
    bars = ax.bar(index, proportions, width=0.95, color='black', label='Actual Proportions')

    for bar in bars:
        y_pos = bar.get_height()
        x_pos = bar.get_x() + bar.get_width()/2
        label = '{:0.2f}'.format(y_pos)
        ax.text(x_pos, y_pos, label, ha='center', va='bottom')

    ax.plot(index, BENFORD_PROPS, c='red', zorder=2, label='Benford Proportions')

    ax.legend()
    
    plt.show()


def main(data_file, p_value=0.05, plot=True):
    series = load_data(data_file)
    actuals = count_first_digits(series)
    expected = get_expected_counts(sum(actuals))
    
    chi2, p = chi_squared_test(actuals, expected)
    print('Chi-Squared Test Statistic: [{:.2f}]\tp-value: [{:.5f}]'.format(chi2, p))
    plot_results(actuals)


if __name__ == '__main__':
    
    data_dir = r'.\data'
    for data_file in os.listdir(data_dir):
        print('\n\nTesting:', data_file)
        file = os.path.join(data_dir, data_file)
        try:
            main(file)
        except Exception as e:
            print(e)