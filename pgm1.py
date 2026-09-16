import pandas as pd
import numpy as np

# Read the dataset
data = pd.read_csv(r"C:\Users\HP\Desktop\ML Syllabus\2.csv")

# Display the dataset
print(data)

# Extract concepts (all columns except the last column)
concepts = np.array(data)[:, :-1]

# Extract target (last column)
target = np.array(data)[:, -1]


# FIND-S algorithm
def train(con, tar):

    # Find the first positive example
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = con[i].copy()
            break

    # Compare all positive examples
    for i, val in enumerate(con):

        # Consider only positive examples
        if tar[i] == 'yes':

            # Compare each attribute
            for x in range(len(specific_h)):

                # If values are different, generalize using '?'
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'

    return specific_h


# Train the model and print the final hypothesis
print(train(concepts, target))

# output: ['sunny' 'warm' '?' 'strong' '?' '?']
