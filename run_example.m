%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an example to test the IMP procedure.
% replace IMP_training() by IMP_inv_training() for the test of IMP_inv 


clear;
clc;

Ttest = 10;   % numer of simulated data sets

d=5+ 1;   % number of features + 1  
n0=300; % sample size for each environment
nu=5;   % number of environments

while Ttest>0

[Y,X,U, A,yi,pu,X_int] = training_data(d,n0,nu);   % generate training data, interve. on both X & Y
[Y_t,X_t,U_t] = testing_data(d,n0,nu,A,pu,yi,X_int);   % generate test data, interve. on both X & Y

[tbl_ind,b_list] = IMP_training(X,Y,U,0.05,0.05);  % training
Y_t_hat  = IMP_testing(tbl_ind,b_list,X_t,U_t);  % testing

res_our(Ttest) = mean((Y_t-Y_t_hat).^2);   % compute rss error for IMP
res_ols(Ttest) = mean((Y_t-[X_t,ones(size(X,1),1)]*ols(Y,[X,ones(size(X,1),1)])).^2); % compute rss error for OLS

Ttest=Ttest-1;

end









