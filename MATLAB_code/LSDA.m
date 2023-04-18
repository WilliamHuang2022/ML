function [A]=LSDA(X,alpha,k,p,tranum)%k近邻,降到p维
%输入列向量
%min \sum_{ij}(a^T y_i - a^T y_j)^2W_{w,ij}
%max \sum_{ij}(a^T y_i - a^T y_j)^2W_{b,ij}
[n,m]=size(X);
Wb=zeros(m,m);
Ww=zeros(m,m);
d_matrix=zeros(m,m);
label=[];
for i=1:m/tranum
    label=[label i*ones(1,tranum)];
end

for i=1:m
    for j=1:m
        d_matrix(i,j)=norm(X(:,i)-X(:,j));
    end
end

for i=1:m
    positions=label~=ceil(i/tranum);
    ds=d_matrix(i,positions);
    [~,ind]=sort(ds);
    values=ds(1,ind(1:k));
    for j=1:k
        site=d_matrix(i,:)==values(j);
        Wb(i,site)=1;
    end
end

for i=1:m
    positions=label==ceil(i/tranum);
    ds=d_matrix(i,positions);
    [~,ind]=sort(ds);
    values=ds(1,ind(2:(k+1)));
    for j=1:k
        site=d_matrix(i,:)==values(j);
        Ww(i,site)=1;
    end
end

Dw=diag(sum(Ww,1));
Db=diag(sum(Wb,1));
Lb=Db-Wb;
[V,D]=eig(X*(alpha*Lb+(1-alpha)*Ww)*X'+0.01*eye(size(X,1)),X*Dw*X'+0.01*eye(size(X,1)));
D=real(D);
[result,ind]=sort(diag(D),'descend');
V=V(:,ind);
A=V(:,sum(isinf(result))+1:p+sum(isinf(result)));

end
