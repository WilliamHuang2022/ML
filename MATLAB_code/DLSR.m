function [W,t]=DLSR(X,lambda,tralabel)
%% ||XW+et'-Y-B@M||_F^2+lambda||W||_F^2
% X is better to be a row-vector matrix
if size(X,1)==size(X,2)
    warning('Plesae ensure the data matrix is a row-vector matrix.')
end
if size(X,1)~=length(tralabel)
    X=X';
end
times_total=10;%迭代次数
[n,m]=size(X);
c=length(unique(tralabel));
%% Y
Y=zeros(n,c);
for i=1:n
    Y(i,tralabel(i))=1;
end
%% B
B=zeros(n,c);
B(Y==1)=1;
B(Y==0)=-1;
%% M
M=rand(n,c);
%% iteration
en=ones(n,1);
H=eye(n)-en*en'/n;
U=(X'*H*X+lambda*eye(m))^(-1)*X'*H;
for time=1:times_total
    R=Y+B.*M;
    W=U*R;
    t=R'*en/n-W'*X'*en/n;
    P=X*W+en*t'-Y;
    M=max(B.*P,0);
end

end