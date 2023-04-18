function [W]=URAFS(X,lambda,alpha,beta,tralabel)
% X is better to be a column-vector matrix.
if size(X,2)~=length(tralabel)
    X=X';
end
epsilon=1e-6;
W_time=8;
F_time=8;
times_total=10;%迭代次数

[d,n]=size(X);
c=length(unique(tralabel));
%% iteration
H=(eye(n)-ones(n,1)*ones(n,1)'/n);
St=X*H*X';
F=orth(rand(n,c));
% S
S=zeros(n);
for i=1:n
    Si_temp=zeros(1,n);
    for j=1:n
        Si_temp(j)=exp(-norm(F(i,:)-F(j,:))^2/2/beta);
    end
    S(i,:)=Si_temp/sum(Si_temp);
end
D=eye(d);
for time=1:times_total
    % Update W
    for Wtime=1:W_time
        B=(St+lambda*D)^(-1/2)*X*H*F;
        [U,~,V]=svd(B,'econ');
        Q=U*V';
        W=(St+lambda*D)^(-1/2)*Q;
        for i=1:d
            D(i,i)=1/(2*sqrt(norm(W(i,:))^2+epsilon));
        end
    end

    % Ls
    P=zeros(n,n);
    for j=1:n
        P(j,j)=(sum(S(j,:))+sum(S(:,j)))/2;
    end
    Ls=P-(S'+S)/2;

    A=H+2*alpha*Ls;
    C=H*X'*W;

    % F
    v=max(max(eig(A)),0)+10;
    Atilde=v*eye(n)-A;
    for Ftime=1:F_time
        R=2*Atilde*F+2*C;
        [U1,~,V1]=svd(R,'econ');
        F=U1*V1';
    end

    % S
    S=zeros(n);
    for i=1:n
        Si_temp=zeros(1,n);
        for j=1:n
            Si_temp(j)=exp(-norm(F(i,:)-F(j,:))^2/2/beta);
        end
        S(i,:)=Si_temp/sum(Si_temp);
    end

end
end