######################################################
# Python Basics
######################################################
import re
from datetime import datetime


def find_vowels_consonants(string):
    """
    Write a function that takes a string in English as input 
    and returns the number of vowels and consonants.
    For example:
    If the string is 'The cat is sleeping', 
    then the function should return : (6, 10)
    """
    # write your code here
    all_char = re.findall("[a-zA-Z]", string)
    vowels_char = re.findall("[aeiou]", string)

    number_vowels = len(vowels_char)
    number_consonants = len(all_char) - len(vowels_char)

    return number_vowels, number_consonants


def make_team():
    """
    Write a function that takes a dictionary called people as input (see below) 
    and returns 3 lists named cook, gardener and clerk containing the name of 
    each individual with this job.
    """

    people = {
        'Pete': {
            'Age': 51,
            'Job': 'Cook'
        },
        'John': {
            'Age': 32,
            'Job': 'Gardener'
        },
        'Jim': {
            'Age': 45,
            'Job': 'Cook'
        },
        'Sheila': {
            'Age': 19,
            'Job': 'Clerk'
        },
        'Carol': {
            'Age': 67,
            'Job': 'Gardener'
        },
        'Richard': {
            'Age': 17,
            'Job': 'Clerk'
        }
    }

    # write your code here
    job_dict = {}
    for key in people.keys():
        job_dict[key] = people[key].get('Job')

    cook = [k for k, v in job_dict.items() if v == 'Cook']
    gardener = [k for k, v in job_dict.items() if v == 'Gardener']
    clerk = [k for k, v in job_dict.items() if v == 'Clerk']

    return cook, gardener, clerk


def find_fridays(start_date, end_date):
    """
    Write a function that given a start date and end date returns how many 
    Fridays there are between the two dates.
    start_date and end_date should be 'YYYY-MM-DD' strings, e.g. '2014-01-31'
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if end_date < start_date:
        return 0

    number_of_friday = int((end_date - start_date).days / 7)

    # count friday in the left days.
    if start_date.weekday() == 4 or end_date.weekday() == 4:
        return number_of_friday + 1

    if start_date.weekday() < 4 and end_date.weekday() > 4:
        return number_of_friday + 1

    if start_date.weekday() == 6 and end_date.weekday() == 5:
        return number_of_friday + 1

    return number_of_friday


def check_number_anagram(integer):
    """
    Write a function that gets an integer number as input and returns a boolean 
    type True or False based on whether the number when read backwards gives 
    the same number. This would be the equivalent of an anagram with strings.
    E.g. input of number 1234321 would return True.
    """

    is_negative = -1 if (integer < 0) else 1

    n = is_negative * int(str(abs(integer))[::-1])

    flag = True if (integer == n) else False
    return flag


if __name__ == '__main__':
    ######################################################
    # Coding test
    ######################################################
    # test find_vowels_consonants
    testString = 'The cat is sleeping'
    [vol, con] = find_vowels_consonants(testString)
    assert (vol == 6 and con == 10)

    # test make_team
    [cook, gardener, clerk] = make_team()
    assert (sorted(cook) == ['Jim', 'Pete'])
    assert (sorted(gardener) == ['Carol', 'John'])
    assert (sorted(clerk) == ['Richard', 'Sheila'])

    # test find_fridays
    assert (find_fridays("2021-03-18", "2021-03-18") == 0)
    assert (find_fridays("2021-03-19", "2021-03-19") == 1)
    assert (find_fridays("2021-03-05", "2021-03-19") == 3)
    assert (find_fridays("2021-03-01", "2021-03-19") == 3)
    assert (find_fridays("2021-03-05", "2021-03-18") == 2)
    assert (find_fridays("2021-03-07", "2021-03-20") == 2)

    # test check_number_anagram
    assert (check_number_anagram(1234321) == True)
    assert (check_number_anagram(0) == True)
    assert (check_number_anagram(-110011) == True)
