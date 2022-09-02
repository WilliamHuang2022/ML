function [Q]=RSLDA(X,tranum,lambda1,lambda2)%列放样本
% $$\min _{P, Q, E} Tr(Q^{T}(S_{w}-uS_{b})Q)+\lambda_{1}\|Q\|_{2,1}+\lambda_{2}\|E\|_{1}$$
% $$\text { s.t. } X=PQ^{T} X+E, P^{T} P=I$$

m=size(X,1);
n=size(X,2);
Sb=zeros(m,m);
Sw=zeros(m,m);
mu_matrix=zeros(m,n/tranum);
for i=1:n/tranum
    mu_matrix(:,i)=mean(X(:,((i-1)*tranum+1):i*tranum),2);
end
mu=mean(X,2);
for i=1:n/tranum
    Sb=Sb+tranum/n*(mu_matrix(:,i)-mu)*(mu_matrix(:,i)-mu)';
end
for i=1:n/tranum
    ui=mu_matrix(:,i);
    for j=1:tranum
        Sw=Sw+(X(:,(i-1)*tranum+j)-ui)*(X(:,(i-1)*tranum+j)-ui)'/n;
    end
end
%初始化
u=1e-4;
beta=0.1;
beta_max=1e5;
rou=1.01;
Q=ones(m,n/tranum);
v=sqrt(sum(Q.*Q,2)+eps);
D=diag(0.5./(v));
E=zeros(m,n);
Y=zeros(m,n);

[P,D_value]=eig(Sw-u*Sb);
D_value=double(D_value);
[y,ind]=sort(diag(D_value));
P=P(:,ind);
P=P(:,1:n);%选择C个特征向量

%迭代
for t=1:20
    % Q
    beta=min(rou*beta,beta_max);
    Q=(2*(Sw-u*Sb)+lambda1*D+beta*X*X')^(-1)*X*(X-E+Y/beta)'*P*beta;
    v=sqrt(sum(Q.*Q,2)+eps);
    D=diag(1./(v));
    % P
    [U,S,V]=svd(X-E+Y/beta,'econ');
    P=U*V';
    % E
    eps1=lambda2/beta;
    temp_E=X-P*Q'*X+Y/beta;
    E=max(0,temp_E-eps1)+min(0,temp_E+eps1);
    % Y
    Y=Y+beta*(X-P*Q'*X-E);
    % beta
    beta=min(rou*beta,beta_max);
end
end