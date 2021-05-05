import pandas as pd


def load_data():
    return pd.read_csv('dataset.csv')


def find_most_common_post_type():
    """
    Write a function that returns the most common type of post (string).
    """
    df = load_data()

    # write your code here
    return df.groupby('Type').size().idxmax()


def is_average_for_paid_higher():
    """
    Write a function that determines if the average of page total likes is 
    higher for paid posts. The function should return a boolean (True / False).
    """
    df = load_data()

    # write your code here
    total_likes_on_cost = df.groupby('Paid')['Page total likes'].mean().idxmax()

    return True if (total_likes_on_cost == 1.0) else False


def determine_best_month_per_category():
    """
    Write a function that returns a list of post months (integers between 1 to 12) 
    that have the highest page total likes for each Category (1, 2, 3 in order).

    """
    df = load_data()

    # write your code here
    return_list = []
    category_list = sorted(df['Category'].unique())
    for category in category_list:
        df_month_per_category = df[df['Category'] == category].groupby(['Post Month'])[
            'Page total likes'].max().idxmax()
        return_list.append(df_month_per_category)

    return return_list

if __name__ == '__main__':
    ######################################################
    # Coding test
    ######################################################
    # test find_most_common_post_type
    type_max = find_most_common_post_type()
    assert (type_max == 'Photo')

    # test is_average_for_paid_higher
    assert (is_average_for_paid_higher() == True)

    # test determine_best_month_per_category
    assert (determine_best_month_per_category() == [12, 12, 12])