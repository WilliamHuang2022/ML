clc,clear

load("Yale5040165.mat")
data_X=Yale5040165;

% load("AR120p20s50by40.mat");
% data_X=AR120p20s50by40;

% load("Pose091632p24s32by32.mat");
% data_X=Pose09_32by32;

% load("FERET74040.mat");
% data_X=FERET74040;

% load("COIL20.mat");
% data_X=COIL20;

%% 此脚本目的在于将算法在用PCA降维到prin后,各个维度下的准确率求出
tranum=3;
tesnum=8;
prin=300;
lambda1_list=-5:5;
lambda2_list=-5:5;
p_list=60;
%% 
[X,testX]=data_divider(data_X,tranum,tesnum);
best_acc=zeros(1,length(p_list));
for ip=1:length(p_list)
    p=p_list(ip);
    %PCA处理=======
    P=PCA(X,prin);
    Xpca=P'*X;
    testXpca=P'*testX;
    record=[];
    for la1=1:length(lambda1_list)
        for la2=1:length(lambda2_list)
            lambda1=lambda1_list(la1);
            lambda2=lambda2_list(la2);
            Q=RSLDA(Xpca,tranum,10^lambda1,10^lambda2);
            Q=real(Q);
            acc=100*KNN_classfier(Q'*Xpca,Q'*testXpca,tranum,tesnum,1);
            record=[record; lambda1 lambda2 acc];
        end
    end
    %==============
    best_acc(ip)=max(record(:,3));
end