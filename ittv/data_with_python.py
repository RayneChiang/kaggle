import pandas as pd


def load_data():
    return pd.read_csv('dataset.csv')


def find_most_common_post_type():
    """
    Write a function that returns the most common type of post (string).
    """
    df = load_data()
    
    # write your code here


def is_average_for_paid_higher():
    """
    Write a function that determines if the average of page total likes is 
    higher for paid posts. The function should return a boolean (True / False).
    """
    df = load_data()
    
    # write your code here


def determine_best_month_per_category():
    """
    Write a function that returns a list of post months (integers between 1 to 12) 
    that have the highest page total likes for each Category (1, 2, 3 in order).

    """
    df = load_data()
    
    # write your code here
