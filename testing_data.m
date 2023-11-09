function  [Y,X,U] = testing_data_exp1_1(d,n0,nu, A,pu,yi,X_int)

n=n0*nu;


U =  repmat(((nu+1):(2*nu)),n0,1);
U = U(:);

var_n = 1;
N = mvnrnd(zeros(d,1),diag(ones(d,1)*var_n),n);

au_m = zeros(d,1);
au_m(pu)=1;


% varying coefficents
at =  unifrnd(0,10,[nu,d]).*( (unidrnd(2,[nu,d])-1.5)*2);  

mean_all = zeros(size(N));

for ii = 1:d
    if  ii ==yi || length(find(X_int==ii))==1
        m_ii = unifrnd(0,10,[nu,1]).*( (unidrnd(2,[nu,1])-1.5)*2);
        mean_ii = repmat(m_ii',n0,1);
        mean_ii = mean_ii(:);
        mean_all(:,ii) =  mean_ii; 
    end
end

N = N+ mean_all;


for j = 1:n

 %   Aj =A  + au_m.*at(U(j)-nu);
    Aj = A;
    Aj(yi,:)= Aj(yi,:)+ au_m' .*at(U(j)-nu,:);

    X1(j,:) = N(j,:)*(eye(d)-Aj')^(-1); 

    
end


Y = X1(:,yi);

X1(:,yi) = [];

X = X1;



end


