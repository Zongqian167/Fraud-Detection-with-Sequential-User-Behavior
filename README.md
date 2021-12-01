# Fraud Detection with Sequential User Behavior

## Background
One of the reasons why using sequential data is so interesting is that people usually focus on the users’ basic information like gender, age, income, family background, and their application date in the  previous fraud detection analysis. However, some missing factors should also be considered which can be hard to evaluate before but now more and more lending is happening online and the development of Deep Learning techniques can be implemented making the evaluation possible. 

## Data Description
In this project, we have the training dataset and testing dataset, which was provided by the  supervisor. In the training dataset, there are 200000 records of user behaviors data of online loan  applications, while in the testing dataset, there are 30000 such records. Both the two datasets have  the same types of data which showed in the following table. 

#### 1. Basic information features (Non-sequential features)  

| Features | Description |
|:--:|:--:|
| label | Reflect if the application defaults or not |
| overdue | Days the loan overdues |
| new_client | Reflect if the applicant is new or not |
| over_time | The submission time of application |

#### 2. Page view behavior features (Sequential features) 

| Features | Description |
|:--:|:--:|
| pname | Name of the current page |
| pstime | Starting viewing time on this page  |
| petime | Ending viewing time on this page |
| pid | Process id |
| sid | Session id |

## Data Processing

After examining both the  training dataset and testing dataset, we found that the only difference is  that the labels, i.e. whether this application went default or not, are provided in the training  dataset but not the testing dataset. However, by observation, we find out that in the training  dataset, if ‘overdue’ > 5, the label of that record corresponds to 1, and 0 otherwise. We apply this  to get the labels for the testing dataset. Note that this is not implemented in the original dataset,  so if you are missing the label for the testing data, you may safely undertake the same method to  generate the labels. 

Preprocessing  is necessary since the data cannot be easily fed into deep learning models in its raw format.

More details can be found in Data_processing.ipynb. Here we mainly used the result from the former group but also made small changes since we find it hard to further improve it.

#### 1. Basic information (Non-sequential information)

Since there exists some missing values which might influence our final performance and the amount is small, we’ve dropped them in our preprocessing part, with reference to the results of previous groups. We revamp the non-sequential features a  little bit to keep the index and format in line with our sequential features. We also extract the day of the week and time of the day at which each application was submitted. 

#### 2. Page viewing behavior (Sequential information)

For each application, we now have a sequence of words that represent the sequential behavior  during applications. In order to keep the hidden information in the sequence behavior, we  concatenate all the words during the same application together with a white space in between  and form lines of words, which we call sequences. Now we have built up our own language  model, where each application is represented as a sequence of words. 

/image/Figure 3 Label Sequence to be fed into SGT model.png
