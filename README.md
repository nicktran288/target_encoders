# Target Encoders

Classes to perform various methods of target encoding. 


### Contents

- [General Information](#general-info)
- [Features](#features)
- [Technologies](#tech)
- [License](#license)


### General Information
Developed to generalize commonly used methods to perform mean/target encoding:
- cvmeanencoder: encodes target mean using kfold scheme where one fold is encoded using out-of-fold samples
- smoothmeanencoder: calculates mean encoding using a parameter to smooth the class and the higher-level/global mean
- twodmeanencoder: calculates mean encoding as a cross between two variables, with one being scaled by the other 


### Features

##### cvmeanencoder
- For train data, encodes using out of fold means; for test data, encodes using the mean of train encodings
- Accepts an arbitrary hierarchy of classes (eg. region, state, city) whereby higher levels will be used depending on
min_samples parameter


##### smoothmeanencoder:
- Creates encodings whereby calculated means are weighted by their sample size and averaged with the higher-level mean
weighted by a specified parameter
- Accepts an arbitrary hierarchy of classes (eg. region, state, city) whereby the mean is smoothed across all levels


##### twodmeanencoder:
- Creates mean encodings which remove the effect of one Class A from Class B, creating a standardized ratio across all
levels of Class A representing the relative impacts of Class B


### Technologies
Built with Python 3.7

##### Uses the following libraries:
- warnings
- numpy
- pandas
- sklearn


### License
MIT 2019