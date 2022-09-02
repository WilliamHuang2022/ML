function [W]=RFS(X,gamma,tranum)
[d,n]=size(X);
c=n/tranum;
m=n+d;

A=[X' gamma*eye(n)];
Y=zeros(n,c);
for i=1:c
    Y(1+(i-1)*tranum:i*tranum,i)=1;
end

D=eye(m);
for t=1:20
    U=D^(-1)*A'*(A*D^(-1)*A')^(-1)*Y;
    D_value=zeros(1,m);
    for i=1:m
        D_value(i)=1/(2*norm(U(i,:)));
    end
    D=diag(D_value);
W=U(1:d,:);
end