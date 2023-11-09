# Invariant-Matching-Property

Data Generation:

Generate data for Experiment A-3 from [1].

training_data.m: generate training data 
testing_data.m: generate test data 



Algorithm:

IMP_inv_training.m: The IMP procedure for training.  
IMP_training.m: The IMP_inv procedure for training.  
IMP_testing.m: The testing algorithm for IMP and IMP_inv. 
run_example.m: Run examples of Experiment A-3.



Utility Functions:


ols.m: Compute OLS estimator.
multi_ols.m: Compute OLS for each environment. 
residual_test.m: Test the invariance of the residual (procedure II from [2]). 


[1] Du, Kang, and Yu Xiang. "Learning invariant representations under general interventions on the response." arXiv preprint arXiv:2208.10027 (2022).

[2] Peters, Jonas, Peter Bühlmann, and Nicolai Meinshausen. "Causal inference by using invariant prediction: identification and confidence intervals." Journal of the Royal Statistical Society Series B: Statistical Methodology 78.5 (2016): 947-1012.

