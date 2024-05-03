# 4th Year Project Proposal

## SECTION A


| Project Title: | Lower Limb Sports Injury Prediction and Prevention |

| Student 1 Name: | Dylan Bagnall |

| Student 1 ID: | 20413066 |

| Student 2 Name: | Lorcan Dunne |

| Student 2 ID: | 19311511 |

| Project Supervisor: | Mark Roantree |

## SECTION B

> Guidance: This document is expected to be approximately 3 pages in length, but it can exceed this page limit. It is also permissible to carry forward content from this proposal to your later documents (e.g. functional specification) as appropriate.
> 
> 
> Your proposal must include *at least* the following sections.
> 

### Introduction

Lower limb injuries are a major and pressing problem in the high intensity sport. These injuries can have a significant impact on players' careers and lives. There is a growing interest in using machine learning to predict and prevent injuries. Machine learning models can be trained on historical data to identify patterns that are associated with injuries. These patterns can then be used to predict which players are at increased risk of injury. This information can then be used to develop prevention strategies.

### Outline

The aim of this project is to develop an AI prediction model that can predict lower limb injuries in the NFL using decision trees, random forest, and XGBoost algorithms. We will use a dataset from the NFL of 105 lower limb injuries and 267,005 player-plays that occurred during the regular season over the two seasons. 

We will follow these steps to develop our AI prediction model:

**Data pre-processing:** We will clean and prepare the NFL dataset for training our model. This may involve removing incomplete or erroneous data, and converting categorical variables to numerical variables.

**Feature engineering:** We may create new features from the existing data or transform the existing data in a way that makes it more informative for our model.

**Model selection:** We will select the machine learning algorithm that is most appropriate for our task. We will consider factors such as the type of data we have, the complexity of the problem, and the computational resources available to us.

**Model training:** We will train our model on the NFL dataset. We will use a variety of hyperparameters to tune the performance of our model.

**Model evaluation:** We will evaluate the performance of our model on a held-out test set. This will give us an estimate of how well our model will perform on new data.

**Model deployment:** Once we are satisfied with the performance of our model, we will deploy it to production. This may involve saving the model to a file or integrating it into a software application using Flask.

### Background

> Where did the ideas come from?
> 
- We initially were thinking of making a predictive model for a Kaggle competition on housing prices.
- We decided to alter this idea to something more in our field of interest once we saw this project was undertaken by students in a previous year.
- Sports and biology is something we both are interested in, so we decided to go down the route of injuries and the factors that cause them.

### Achievements

> What functions will the project provide? Who will the users be?
> 

Our AI prediction model will be able to accurately predict lower limb injuries in high intensity sports. This information can then be used by players and coaches to develop prevention strategies.

### Justification

> Why/when/where/how will it be useful?
> 

Our AI prediction model will prove to be extremely beneficial and advantageous for both players and coaches in high intensity sports. they will have the opportunity to not only enhance their overall performance but also significantly minimize the occurrence of lower limb injuries. Our model will empower them with valuable insights and data-driven strategies, enabling them to proactively develop effective prevention techniques and foster a safer and more secure environment for all athletes involved.

### Programming language(s)

> List the proposed language(s) to be used.
> 

We plan to use Python to develop our AI predictors model.

### Programming tools / Tech stack

> Describe the compiler, database, web server, etc., and any other software tools you plan to use.
> 

We will use the following programming tools and tech stack to develop our AI prediction model:

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Flask

### Hardware

> Describe any non-standard hardware components which will be required.
> 
- We will not require any non-standard hardware components to complete our predictive model.

### Learning Challenges

> List the main new things (technologies, languages, tools, etc) that you will have to learn.
> 
- The main learning challenges we will face are using new tools, such as;
    - Flask/Scikit-learn/Matplotlib
- Learning and applying machine learning algorithms:
    - each algorithm is different so finding and applying suitable ones will be a challenge
- Tuning the model:
    - When we have chosen the algorithm that we want to use, we need to tune the models parameters to get the best possible performance
- Transfer learning:
    - We will need to use what we learned from our trained model on the NFL to create a predicitive model on the GAA and Rugby where the datasets are not as large.
- Interpreting and evaluating the results from our model:
    - We have no previous experience in interpreting predictions from a machine learning model

### Breakdown of work

> Clearly identify who will undertake which parts of the project.
> 
> 
> It must be clear from the explanation of this breakdown of work both that each student is responsible for separate, clearly-defined tasks, and that those responsibilities substantially cover all of the work required f  or the project.
> 

While we both will have individual tasks and responsibilities to train this model, we will plan on completing the majority of tasks together as a pair. We feel that working as a pair will prove much more productive and the tasks will be completed at a higher standard.

### Student 1

> Student 1 should complete this section.
> 
- Data pre-processing
- Decision tree
- Random forest algorithm
- Data evaluation and interpretation from random forest algorithm results
- Model deployment using Flask

### Student 2

> Student 2 should complete this section.
> 
- Feature Engineering
- Decision tree
- XGBoost
- Data evaluation and interpretation from XGBoost algorithm results
- Model deployment using Flask
