import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import warnings
warnings.filterwarnings('ignore')

SEED = 42

def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)


def get_pclass_dist(df):

        # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]

    for deck in decks:
        for pclass in range(1, 4):
            try:
                    count = df[deck][pclass][0]
                    deck_counts[deck][pclass] = count
            except KeyError:
                    deck_counts[deck][pclass] = 0

    df_decks = pd.DataFrame(deck_counts)
    deck_percentages = {}

    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
            deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]

    return deck_counts, deck_percentages

def display_pclass_dist(percentages):

        df_percentages = pd.DataFrame(percentages).transpose()
        deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
        bar_count = np.arange(len(deck_names))
        bar_width = 0.85

        pclass1 = df_percentages[0]
        pclass2 = df_percentages[1]
        pclass3 = df_percentages[2]

        plt.figure(figsize=(20, 10))
        plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')
        plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width,
                label='Passenger Class 2')
        plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width,
                label='Passenger Class 3')

        plt.xlabel('Deck', size=15, labelpad=20)
        plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)
        plt.xticks(bar_count, deck_names)
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
        plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)

        plt.show()


if __name__ == '__main__':
    test_data = pd.read_csv("test.csv")
    train_data = pd.read_csv("train.csv")
    sample_data = pd.read_csv("gender_submission.csv")

    all_data = concat_df(train_data, test_data)

    train_data.name = 'Training Set'
    test_data.name = 'Test Set'
    all_data.name = 'All Set'

    dfs = [train_data, test_data]
    for df in dfs:
        print('{}'.format(df.name))
        display_missing(df)


    print('Number of Training Examples = {}'.format(train_data.shape[0]))
    print('Number of Test Examples = {}\n'.format(test_data.shape[0]))
    print('Training X Shape = {}'.format(train_data.shape))
    print('Training y Shape = {}\n'.format(train_data['Survived'].shape[0]))
    print('Test X Shape = {}'.format(test_data.shape))
    print('Test y Shape = {}\n'.format(test_data.shape[0]))
    print(test_data.columns)
    print(train_data.columns)
    print(train_data.info())

    # the correlation between features
    df_all_corr = all_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'},
                       inplace=True)
    correlation = df_all_corr[df_all_corr['Feature 1'] == 'Survived']

    age_by_pclass_sex = all_data.groupby(['Sex', 'Pclass']).median()['Age']

    # for pclass in range(1, 4):
    #     for sex in ['female', 'male']:
    #         print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
    # print('Median age of all passengers: {}'.format(all_data['Age'].median()))

    # Filling the missing values in Age with the medians of Sex and Pclass groups
    all_data['Age'] = all_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    all_data[all_data['Embarked'].isnull()]
    # Filling the missing values in Embarked with S
    all_data['Embarked'] = all_data['Embarked'].fillna('S')
    med_fare = all_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    # Filling the missing value in Fare with the median Fare of 3rd class alone passenger
    all_data['Fare'] = all_data['Fare'].fillna(med_fare)

    # Creating Deck column from the first letter of the Cabin column (M stands for Missing)
    all_data['Deck'] = all_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

    df_all_decks = all_data.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch',
                                                                            'Fare', 'Embarked', 'Cabin', 'PassengerId',
                                                                            'Ticket']).rename(
        columns={'Name': 'Count'}).transpose()


    all_deck_count, all_deck_per = get_pclass_dist(df_all_decks)
    # display_pclass_dist(all_deck_per)


    idx = all_data[all_data['Deck'] == 'T'].index
    all_data.loc[idx, 'Deck'] = 'A'

    all_data['Deck'] = all_data['Deck'].replace(['A', 'B', 'C'], 'ABC')
    all_data['Deck'] = all_data['Deck'].replace(['D', 'E'], 'DE')
    all_data['Deck'] = all_data['Deck'].replace(['F', 'G'], 'FG')

    print(all_data['Deck'].value_counts())

    # Dropping the Cabin feature
    all_data.drop(['Cabin'], inplace=True, axis=1)

    df_train, df_test = divide_df(all_data)
    dfs = [df_train, df_test]

    for df in dfs:
        display_missing(df)


    # target distribution

    survived = df_train['Survived'].value_counts()[1]
    not_survived = df_train['Survived'].value_counts()[0]
    survived_per = survived / df_train.shape[0] * 100
    not_survived_per = not_survived / df_train.shape[0] * 100

    print('{} of {} passengers survived and it is the {:.2f}% of the training set.'.format(survived, df_train.shape[0],
                                                                                           survived_per))
    print('{} of {} passengers didnt survive and it is the {:.2f}% of the training set.'.format(not_survived,
                                                                                                df_train.shape[0],
                                                                                                not_survived_per))

    plt.figure(figsize=(10, 8))
    sns.countplot(df_train['Survived'])

    plt.xlabel('Survival', size=15, labelpad=15)
    plt.ylabel('Passenger Count', size=15, labelpad=15)
    plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)

    plt.title('Training Set Survival Distribution', size=15, y=1.05)

    plt.show()

    # correlations
    df_train_corr = df_train.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(kind="quicksort",
                                                                                              ascending=False).reset_index()
    df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'},
                         inplace=True)
    df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
    df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

    df_test_corr = df_test.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'},
                        inplace=True)
    df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
    df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)

    # Training set high correlations
    corr = df_train_corr_nd['Correlation Coefficient'] > 0.1
    df_train_corr_nd[corr]

    fig, axs = plt.subplots(nrows=2, figsize=(20, 20))

    sns.heatmap(df_train.drop(['PassengerId'], axis=1).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm',
                annot_kws={'size': 14})
    sns.heatmap(df_test.drop(['PassengerId'], axis=1).corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm',
                annot_kws={'size': 14})

    for i in range(2):
        axs[i].tick_params(axis='x', labelsize=14)
        axs[i].tick_params(axis='y', labelsize=14)

    axs[0].set_title('Training Set Correlations', size=15)
    axs[1].set_title('Test Set Correlations', size=15)

    plt.show()


    # target distribution in features
    cont_features = ['Age', 'Fare']
    surv = df_train['Survived'] == 1

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
    plt.subplots_adjust(right=1.5)

    for i, feature in enumerate(cont_features):
        # Distribution of survival in feature
        sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
        sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

        # Distribution of feature in dataset
        sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
        sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

        axs[0][i].set_xlabel('')
        axs[1][i].set_xlabel('')

        for j in range(2):
            axs[i][j].tick_params(axis='x', labelsize=20)
            axs[i][j].tick_params(axis='y', labelsize=20)

        axs[0][i].legend(loc='upper right', prop={'size': 20})
        axs[1][i].legend(loc='upper right', prop={'size': 20})
        axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

    axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
    axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

    plt.show()