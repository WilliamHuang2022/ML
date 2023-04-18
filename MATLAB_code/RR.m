function [P]=RR(X,lambda,tralabel)
% X is better to be a row-vector matrix.
if size(X,1)~=length(tralabel)
    X=X';
end
[n,d]=size(X);
c=length(unique(tralabel));
%% Y
Y=zeros(n,c);
for i=1:n
    Y(i,tralabel(i))=1;
end

P=([X ones(n,1)]'*[X ones(n,1)]+lambda*eye(d+1))^(-1)*[X ones(n,1)]'*Y;
end