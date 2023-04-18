function projection_matrix=PCA(data_set,k)
%输入的是样本矩阵展开成的列向量再合并而成的矩阵
amount=size(data_set,2);
mean_vector=sum(data_set,2)/amount;
scatter_matrix=zeros(size(data_set,1));
for i=1:amount
    s1=(data_set(:,i)-mean_vector)*(data_set(:,i)-mean_vector)';
    scatter_matrix=scatter_matrix+s1;
end
[X,Y]=eig(scatter_matrix);
Y=double(Y);
[~,ind]=sort(diag(Y),"descend");
X=X(:,ind);
projection_matrix=X(:,1:k);
