# I wrote this code to count how many times unique words are used in a string of text. I originally made it for cover letters, but it can be used for anything where you don't want to repeat a word too many times.

def wordFrequency(text_input):

    # Set of filler words to exclude from the count.
    filler_words = {"and", "to", "for", "but", "a", "the", "in", "on", "at", "with", "is", "of", "or", "as", "an", "be"}

    # Create a dictionary for storing the words.
    storage = {}

    # String for storing the current word.
    word = ""

    for i in range(len(text_input)):
        # Check if the current character is a blank space or a punctuation mark.

        if text_input[i] in {' ', '.', ',', ';', ':', '!', '?', '-', '\n'}:

            # Make sure capitalization is not an issue.

            if word and word.lower() not in filler_words:

                # If the current word is not found, insert the current word with frequency 1. This will be added to if the word repeats.

                if word.lower() not in storage:
                    storage[word.lower()] = 1
                else:
                    storage[word.lower()] += 1
            word = ""

        else:
            word += text_input[i]

    # Storing the last word of the string
    if word and word.lower() not in filler_words:
        if word.lower() not in storage:
            storage[word.lower()] = 1
        else:
            storage[word.lower()] += 1

    # Sort the words from most to least used, then print the results.

    sorted_storage = sorted(storage.items(), key=lambda item: item[1], reverse=True)

    for word, freq in sorted_storage:
        print(word, "-", freq)

text_input = 'One situation that could incorporate decision trees is a bank or other financial institution assessing loan applicants to determine whether they qualify for a loan, which could be a simple Yes/No, in this case of classification. Using decision trees would be one way to break down applicants into these categories based on a variety of collected attributes. This would speed up the approval process; consider more than one or two aspects of an applicant’s financial situation so as to not discriminate or overlook extenuating circumstances; potentially save the institution a lot of money; and give analysts and their coworkers an idea as to which of these attributes matter the most. They can use historical data of people who have and have not defaulted on a loan or made too many late payments to help predict who will or will not cause issues in the future.'
wordFrequency(text_input)

''' Output: one - 3
loan - 3
not - 3
situation - 2
could - 2
decision - 2
trees - 2
financial - 2
institution - 2
applicants - 2
they - 2
which - 2
this - 2
would - 2
these - 2
attributes - 2
who - 2
have - 2
will - 2
that - 1
incorporate - 1
bank - 1
other - 1
assessing - 1
determine - 1
whether - 1
qualify - 1
simple - 1
yes/no - 1
case - 1
classification - 1
using - 1
way - 1
break - 1
down - 1
into - 1
categories - 1
based - 1
variety - 1
collected - 1
speed - 1
up - 1
approval - 1
process - 1
consider - 1
more - 1
than - 1
two - 1
aspects - 1
applicant’s - 1
so - 1
discriminate - 1
overlook - 1
extenuating - 1
circumstances - 1
potentially - 1
save - 1
lot - 1
money - 1
give - 1
analysts - 1
their - 1
coworkers - 1
idea - 1
matter - 1
most - 1
can - 1
use - 1
historical - 1
data - 1
people - 1
defaulted - 1
made - 1
too - 1
many - 1
late - 1
payments - 1
help - 1
predict - 1
cause - 1
issues - 1
future - 1
'''
