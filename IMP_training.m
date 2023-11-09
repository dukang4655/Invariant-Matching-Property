function [tbl_ind,b_list] = IMP_training(X,Y,U,eps,alpha)


[n,d] = size(X);    
subset_s = logical( ff2n(d));
subset_s(1,:) = [];
ns = size(subset_s,1); 

res = ones(d, ns, ns)*100000;    % IMP score 
res_imp = ones(d, ns, ns)*100000;  % prediction score 


% exhaustive seach over (k,R,S)'s
for ik =1:d

    X_k = X(:,ik);

    for is = 1:ns

        X_s = X(:,subset_s(is,:)); 
        size_s = sum(subset_s(is,:)); 
        subset_r = logical( ff2n(size_s));

         subset_r(1,:) = [];

        if subset_s(is,ik)>0   
            wch_k = sum(subset_s(is,1:ik));
                subset_r(subset_r(:,wch_k)==1,:)=[];
        end  

        nr =  size(subset_r,1);
        for ir = 1:nr
            X_r = X_s(:, subset_r(ir,:) );

          feature_X =multi_ols(X_k,X_r,U);  

              Y_hat =   multi_ols(Y,X_s,U);

               b1= ols(Y_hat, [ones(n,1),feature_X,X_s]);
                Y_imp = [ones(n,1),feature_X,X_s]*b1;

                res_imp(ik,is,ir)  = mean((Y_hat- Y_imp).^2);   % compute IMP score
            
                 res(ik,is,ir) =  mean((Y- Y_imp).^2);   % compute prediciton score
 
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute cutoff parameter for IMP score

 all_res = res(:);
 all_res(all_res==100000)=[];
 c0 = prctile(all_res,50);    % median of all prediction scores

res_imp(res>c0) = 100000;

[~,idx]=min(res_imp(:));
[ik,is,ir]  =ind2sub(size(res_imp),idx);

size_s = sum(subset_s(is,:)); 
X_k = X(:,ik);
X_s = X(:,subset_s(is,:)); 

subset_r = logical( ff2n(size_s));

subset_r(1,:) = [];

 
if subset_s(is,ik)>0   
    wch_k = sum(subset_s(is,1:ik));
        subset_r(subset_r(:,wch_k)==1,:)=[];
end  
X_r = X_s(:, subset_r(ir,:) );

B_num = 50;

U_list  = unique(U);

for ii=1:numel(U_list)
 U_size(ii) =   sum(U == U_list(ii));
end


B_size = min(U_size)/3; 

for j = 1: B_num
    for i=1:numel(U_list)
        indx(((i-1)*B_size+1): (i*B_size) ) = datasample(1:U_size(i) ,B_size)+sum(U_size(1:(i-1))) ;
    end
    X_Bk = X_k(indx,:);
    X_Br = X_r(indx,:);  
    X_Bs = X_s(indx,:); 
    U_B = U(indx);
    Y_B = Y(indx);
    
    feature_X =multi_ols(X_Bk,X_Br,U_B);  
    Y_hat_B =   multi_ols(Y_B,X_Bs,U_B);
    
    b= ols( Y_hat_B, [ones(size(X_Bk,1),1),feature_X,X_Bs]);
    Y_pred_B = [ones(size(X_Bk,1),1),feature_X,X_Bs]*b;
    
    B_rss(j) =  mean(( Y_hat_B- Y_pred_B).^2);
end

[f,x] = ecdf(B_rss);

c1 = min(x((1-eps)<=f));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute cutoff parameter for prediction score

res(res_imp>c1) = 100000;

[~,idx]=min(res(:));
[ik,is,ir]  =ind2sub(size(res),idx);

size_s = sum(subset_s(is,:)); 
X_k = X(:,ik);
X_s = X(:,subset_s(is,:)); 

subset_r = logical( ff2n(size_s));

subset_r(1,:) = [];

 
if subset_s(is,ik)>0   
    wch_k = sum(subset_s(is,1:ik));
        subset_r(subset_r(:,wch_k)==1,:)=[];
end  

X_r = X_s(:, subset_r(ir,:) );

B_num = 50;
B_size = min(U_size)/3;

for j = 1: B_num
    for i=1:numel(U_list)
    indx(((i-1)*B_size+1): (i*B_size) ) = datasample(1:U_size(i) ,B_size)+sum(U_size(1:(i-1))) ;
    end
    X_Bk = X_k(indx,:);
    X_Br = X_r(indx,:);  
    X_Bs = X_s(indx,:); 
    U_B = U(indx);
    Y_B = Y(indx);

    feature_X =multi_ols(X_Bk,X_Br,U_B);  
    b= ols(Y_B, [ones(size(X_Bk,1),1),feature_X,X_Bs]);
    Y_pred_B = [ones(size(X_Bk,1),1),feature_X,X_Bs]*b;

   B_rss(j) =  mean((Y_B- Y_pred_B).^2);
end

[f,x] = ecdf(B_rss);

c2 = min(x((1-alpha)<=f));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% return the identified IMPs

idx= find(res<c2);
[ck,cs,cr]  =ind2sub(size(res),idx);

tbl_ind=[ck,cs,cr];

for it=1:size(tbl_ind,1)
    ik =  tbl_ind(it,1);
    is = tbl_ind(it,2);
    ir = tbl_ind(it,3);
    
    size_s = sum(subset_s(is,:)); 
    subset_r = logical( ff2n(size_s));
    subset_r(1,:) = [];

    if subset_s(is,ik)>0   
        wch_k = sum(subset_s(is,1:ik));
            subset_r(subset_r(:,wch_k)==1,:)=[];
    end  
   
        X_s = X(:,subset_s(is,:)); 
        X_k = X(:,ik);

        Y_hat =   multi_ols(Y,X_s,U);

        X_r = X_s(:, subset_r(ir,:) );
        feature_X =multi_ols(X_k,X_r,U);  
    
        b_list{it} =  ols(Y_hat, [ones(n,1),feature_X,X_s]);

end

end