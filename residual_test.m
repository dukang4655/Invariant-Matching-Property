function p_val = residual_test(res,U)

U_sp = unique(U);

nu = length(U_sp); 

for ii = 1: nu


    res1 = res(U==U_sp(ii));
    res2 = res(U~= U_sp(ii)); 

     n1 = size(res1,1); 
    n2 = size(res2,1); 

    s1 = sum((res1-mean(res1)).^2)/(n1-1);
    s2 = sum((res2-mean(res2)).^2)/(n2-1);

    Ttest = (mean(res1)-mean(res2))/(s1/n1+s2/n2)^0.5; 

    Ftest = s1/s2; 

    dof = (s1/n1+s2/n2)^2/((s1/n1)^2/(n1-1)+(s2/n2)^2/(n2-1));

    pval1(ii) = (1- tcdf(Ttest,dof))*2; 

    pval2(ii) =1- fcdf(Ftest,n1-1,n2-2); 

end

 p_val1 = min(pval1) *nu; 
  p_val2 = min(pval2) *nu; 
p_val = min(p_val1,p_val2)*2;

end


