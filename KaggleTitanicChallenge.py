import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
#passengers is the data frame 

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female':'1','male':'0'})


# Fill the nan values in the age column
#print(passengers['Age'].values)
#meanAge = passengers['Age'].mean()
passengers['Age'].fillna(value='29.7',inplace=True)



# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)


# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
#print(passengers)

# Select the desired features
features = passengers[['Sex','Age','FirstClass','SecondClass']]

survival = passengers['Survived']

# Perform train, test, split

train_test_split(features,labels,test_size = 0_to_1_test_size)

# Scale the feature data so it has mean = 0 and standard deviation = 1


# Create and train the model


# Score the model on the train data


# Score the model on the test data


# Analyze the coefficients


# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
# You = np.array([___,___,___,___])

# Combine passenger arrays


# Scale the sample passenger features


# Make survival predictions!

