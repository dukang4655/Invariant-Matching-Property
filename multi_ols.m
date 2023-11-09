function Y_hat = multi_ols(Y,X,U)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% environment-wise OLS

U_sp = unique(U);

Y_hat = zeros(size(Y)); 

for ii = 1: length(U_sp)

    pos_u_ii = find(U==U_sp(ii)); 
    X1 = X( pos_u_ii, :);
    X1 = [X1,ones(size(X1,1),1)];
    Y1 = Y( pos_u_ii);
    b_hat1 = inv(X1'*X1)*X1'*Y1; 
    
    Y_hat(pos_u_ii) = X1*b_hat1;

end