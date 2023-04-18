function [U]=CRFS(X,lambda,tranum)
sigma=1;
[d,n]=size(X);
c=n/tranum;
Y=zeros(n,c);
for i=1:c
    Y(1+(i-1)*tranum:i*tranum,i)=1;
end
U=rand(d,c);
eps1=0.1;
for t=1:20
    P_value=zeros(1,n);
    Q_value=zeros(1,d);
    for k=1:n
        P_value(1,k)=exp(-norm(X'*U-Y)^2/sigma^2);
    end
    for i=1:d
        Q_value(1,i)=1/(2*norm(U(i,:)+eps1));
    end
    P=diag(P_value);
    Q=diag(Q_value);
    U=(X*P*X'+lambda*Q)^(-1)*(X*P*Y);

end
