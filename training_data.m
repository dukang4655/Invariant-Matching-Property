function  [Y,X,U, A,yi,pu, X_int] = training_data_exp1_1(d,n0,nu)

n= n0*nu;

U =  repmat((1:nu),n0,1);
U = U(:);

var_n = 1;

N = mvnrnd(zeros(d,1),diag(ones(d,1)*var_n),n);


% random dag,  random select Y,  Y has at least 1 child and 1 parents

while(1)
B = tril(unidrnd(2,[d,d])-1, -1);
yi = unidrnd(d-2)+1;

%n_mb = sum(B(yi,:)) + sum(B(:,yi)) + sum(B(B(:,yi)>0,:), 'all' );
    if sum(B(yi,:))>=1  && sum(B(:,yi))>=1 %&& n_mb>=floor(d/2)
        break; 
    end
end


% coefficient matrix [-1.5, 0.5] U [0.5, 1.5]

coef_B = unifrnd(0.5,1.5,[d,d]) .*( (unidrnd(2,[d,d])-1.5)*2);
A = B.* coef_B;

%g = digraph(B');
%isdag(g)


% random select parents of Y to have varying coefficents
pa = find(B(yi,:)==1);
np = length(pa); 
%n_au = unidrnd(np); 
n_au = np;
comb = nchoosek(pa, n_au);
dice = unidrnd(size(comb,1));
pu = comb(dice,:);
au_m = zeros(d,1);
au_m(pu)=1;

% select intervened predictors
x_ind= ones(d,1);
x_ind(yi)=0;

ch_y = B(:,yi);
n_ch = sum(ch_y);
if n_ch ==1
    x_ind(ch_y >0) = 0;
else
    ch_y_ind = find(ch_y >0); 
    un_int_ch = ch_y_ind(unidrnd(n_ch));
    x_ind(un_int_ch) = 0;
end
%num_int = unidrnd(d-2); 
num_int = 4;
comb = nchoosek(find(x_ind>0), num_int);
ch_one = unidrnd (size(comb,1));
X_int = comb(ch_one,:);

for ii = 1:d
    if  ii ==yi || length(find(X_int==ii))==1
        m_ii = unifrnd(0,2,[nu,1]).*( (unidrnd(2,[nu,1])-1.5)*2);
        mean_ii = repmat(m_ii',n0,1);
        mean_ii = mean_ii(:);
    else
        m_ii =  0;  
        mean_ii = repmat(m_ii,n,1);
    end
        mean_all(:,ii) =  mean_ii; 
end

N = N+ mean_all; 

% varying coefficents

at =  unifrnd(0,4,[nu,d]).*( (unidrnd(2,[nu,d])-1.5)*2);  

% generate the variables
for j = 1:n

    Aj = A;
    Aj(yi,:)= Aj(yi,:)+ au_m' .*at(U(j),:);
    
    X1(j,:) = N(j,:)*(eye(d)-Aj')^(-1);
    
end


Y = X1(:,yi);

X1(:,yi) = [];

X=X1;

end


