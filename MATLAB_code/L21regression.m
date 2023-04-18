dataset=AR120p20s50by40_train;
n=size(dataset,3);

gama=1000;                                                      %gama参数
lamuda=100;                                                   %求逆修正系数
t=15;                                                        %迭代次数
datavectors=matrix2vectors(dataset);
d=size(datavectors,1);
I=ones(1,n);
datavectors1=[datavectors;I];

m=n+d+1;

X=datavectors1;
c=120;                                                           %类数
withinnumber=7;                                                %训练集类内个数
A=[X' gama*eye(n)];
Y=zeros(n,c);
for i=1:c
    Y(1+(i-1)*withinnumber:i*withinnumber,i)=1;
end
D=eye(m);


for i=1:t
    U=inv(D+lamuda*eye(m))*A'*inv(A*inv(D+lamuda*eye(m))*A'+lamuda*eye(n))*Y;
    for j=1:m
        D(m,m)=1/(2*norm(U(j,:)));
    end
end
W=U(1:d+1,:);

accuracy=L21regressionclassifier(AR120p20s50by40_test,W,7)        %修改测试集，类内个数
