function [W,Theta]=FSOR(X,tralabel)
% ||W'*Theta*X+b*en'-Y|| 
% s.t. W'*W=I, theta'*ed=1,theta>=0 (theta is the diagonal element of Theta

% X is better to be a column-vector matrix
if size(X,1)==size(X,2)
    warning('Plesae ensure the data matrix is a column-vector matrix.')
end
if size(X,2)~=length(tralabel)
    X=X';
end

GPI_time=10;
rou=1.5; %ALM参数
ALM_time=10;
times_total=13;%迭代次数
[d,n]=size(X);
c=length(unique(tralabel));

%% Y
Y=zeros(n,c);
for i=1:n
    Y(i,tralabel(i))=1;
end
Y=Y';
%% iteration
theta=rand(d,1);
theta=theta/sum(theta);
Theta=diag(theta);
H=eye(n)-ones(n,1)*ones(n,1)'/n;
W=rand(d,c);
for time=1:times_total
    % W
    A=Theta*X*H*X'*Theta';
    B=Theta*X*H*Y';
    alpha=max(max(eig(A)),0)+10;
    Atilde=alpha*eye(d)-A;
    for gpi=1:GPI_time
        M=2*Atilde*W+2*B;
        [U,~,V]=svd(M,'econ');
        W=U*V';
    end
    
    % Theta
    Q=(X*H*X').*(W*W');
    s=diag(2*X*H*Y'*W');
    theta=ones(d,1)/d;
    v=theta;
    lambda2=0;
    mu=3;
    lambda1=zeros(d,1);
    for alm=1:ALM_time
        E=2*Q+mu*eye(d)+mu*ones(d,1)*ones(1,d);
        f=mu*v+mu*ones(d,1)-lambda2*ones(d,1)-lambda1+s;
        theta=E^(-1)*f;
        v_temp=theta+1/mu*lambda1;
        v_temp(v_temp<0)=0;
        v=v_temp;
        lambda1=lambda1+mu*(theta-v);
        lambda2=lambda2+mu*(theta'*ones(d,1)-1);
        mu=rou*mu;
    end

end
Theta=diag(theta);
end