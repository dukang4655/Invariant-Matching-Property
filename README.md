# Invariant-Matching-Property

Implementation of the `IMP` and `IMP_inv` algorithms from the work 
"Learning invariant representations under general interventions on the response" by Kang Du and Yu Xiang.

>Data Generation: Generate data for Experiment A-3 from [1].

`training_data.m`: Generate training data. 

`testing_data.m`: Generate test data. 



>Algorithms:

`IMP_inv_training.m`: The training procedure of `IMP`.  

`IMP_training.m`: The training procedure of `IMP_inv`.  

`IMP_testing.m`: The testing procedure of `IMP` and `IMP_inv`. 

`run_example.m`: Run examples of Experiment A-3.



>Utility Functions:


`ols.m`: Compute OLS estimator.

`multi_ols.m`: Compute OLS for each environment. 

`residual_test.m`: Test the invariance of the residual (procedure II from [2]). 

>References:

[1] Du, Kang, and Yu Xiang. "Learning invariant representations under general interventions on the response." arXiv preprint arXiv:2208.10027 (2022).

[2] Peters, Jonas, Peter Bühlmann, and Nicolai Meinshausen. "Causal inference by using invariant prediction: identification and confidence intervals." Journal of the Royal Statistical Society Series B: Statistical Methodology 78.5 (2016): 947-1012.

