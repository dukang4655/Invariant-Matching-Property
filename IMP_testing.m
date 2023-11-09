function Y_hat = IMP_testing(tbl_ind,b_list,X,U)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluate the estimated IMPs on test data

[n,d] = size(X); 

nf = size(tbl_ind,1); 
Y_hat = zeros(n,1); 
subset_s = logical( ff2n(d));
subset_s(1,:) = [];

    
    for it=1:nf
        ik =  tbl_ind(it,1);
        is = tbl_ind(it,2);
        ir = tbl_ind(it,3);
        
        X_s = X(:,subset_s(is,:)); 
        
        size_s = sum(subset_s(is,:)); 
        subset_r = logical( ff2n(size_s));
        subset_r(1,:) = [];
        
        if subset_s(is,ik)>0   
            wch_k = sum(subset_s(is,1:ik));
            subset_r(subset_r(:,wch_k)==1,:)=[];
        end  
        
        X_k = X(:,ik);
        
        X_r = X_s(:, subset_r(ir,:) );
        feature_X =multi_ols(X_k,X_r,U);  
        Y_hat_i =  [ones(n,1),feature_X,X_s]*b_list{it};
        
        
        Y_hat = Y_hat + Y_hat_i;
    end


  Y_hat= Y_hat/nf;   % return the averaged prediction model

end