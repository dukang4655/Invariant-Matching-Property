function b_hat = ols(Y,X)

p = size(X,2); 

if rank(X'*X) == size(X,2)
b_hat = inv(X'*X)*X'*Y; 

else
    b_hat = pinv(X'*X)*X'*Y; 
end