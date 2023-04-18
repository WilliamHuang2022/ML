function [W,b]=GReLSR(X,lambda,gamma,tralabel)
% ||XW+en*b'-Y-Y.*U-a*ec'||+lambda*||W||+gamma*R(a,mu)
% s.t. U>=0, U_{i,yi}=0
% X is better to be a row-vector matrix
if size(X,1)==size(X,2)
    warning('Plesae ensure the data matrix is a row-vector matrix.')
end
if size(X,1)~=length(tralabel)
    X=X';
end
times_total=10;%迭代次数
[n,d]=size(X);
c=length(unique(tralabel));
%% Y
Y=-ones(n,c);
for i=1:n
    Y(i,tralabel(i))=1;
end
%% iteration
a=ones(n,1);
H=eye(n)-ones(n,1)*ones(n,1)'/n;
U=-Y;
U(U==-1)=0;
mu=ones(c,1);
for time=1:times_total
    T=Y+Y.*U+a*ones(c,1)';
    % W
    W=(X'*H*X+lambda*eye(d))^(-1)*X'*H*T;
    % b
    b=(T'*ones(n,1)-W'*X'*ones(n,1))/n;

    R=X*W+ones(n,1)*b'-Y;
    % Update U a
    for i=1:n
        yi=tralabel(i);
        muk=mu(tralabel(i));
        r=R(i,:);
        y=Y(i,:);
        u=U(i,:);
        ai=a(i);

        alpha=r(yi)+gamma*muk; beta=1+gamma;
        r=sort(r,'descend');
        for m=1:c
            if m==yi
                continue
            end
            ai=(sum(r(1:(m-1)))+alpha)/(m-1+beta);
            if ai>=r(m)
                break
            end
        end
        u(1:(m-1))=0;
        for j=m:c
            if m==yi
                continue
            end
            u(j)=ai-r(j);
        end
        U(i,:)=u;
        a(i)=ai;
    end
    % Update mu
    for j=1:c
        mu(j)=sum(a(tralabel==j))/sum(tralabel==j);
    end
end

end