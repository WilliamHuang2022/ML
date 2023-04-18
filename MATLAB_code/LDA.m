function projection_matrix=LDA(data_set,kind_number,m)
%输入的是已经展成列向量的数据集！！！
amount=size(data_set,2);
dimension=size(data_set,1);

mean_total=sum(data_set,2)/amount;

Sb=zeros(dimension);
for j=1:(amount/kind_number)
    local_vectors=data_set(:,1+kind_number*(j-1):kind_number*j);
    local_mean=sum(local_vectors,2)/kind_number;
    local_Sb=(local_mean-mean_total)*(local_mean-mean_total)';
    Sb=Sb+local_Sb;
end
 
Sw=zeros(dimension);
for j=1:(amount/kind_number)
    local_vectors=data_set(:,1+kind_number*(j-1):kind_number*j);
    local_mean=sum(local_vectors,2)/kind_number;
    local_Sw=zeros(dimension);
    for k=1:kind_number
        local_Sw=local_Sw+(local_vectors(:,k)-local_mean)*(local_vectors(:,k)-local_mean)';
    end
    Sw=Sw+local_Sw;
end
Sw_improve=Sw+0.01*eye(dimension);

goal_martrix=inv(Sw_improve)*Sb;
[X,Y]=eig(goal_martrix);
[y,ind]=sort(diag(Y));
X=X(:,ind);
projection_matrix=X(:,dimension-m+1:dimension);
