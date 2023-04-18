function [X]=DataDivider(name,proportion,options)
% Input:
% name: the name of the dataset.
% proportion: the proportion of training sample of each class

% Output:
% X.train: Columns of vectors of training data points. Each column is x_i
% X.test: Columns of vectors of testing data points. Each column is x_i

% options sample:
% options=[];
% options.pca=100;
% options.normalize='FaceData01'; % Divide each element by 255
% options.normalize='Unit'; % Transform each sample vector into a unit vector
% options.normalize='zscore';

X=[];

for i=1:1
switch lower(name)
    case lower('ar')
        load("E:\ML\database\AR120p26s50by40.mat")
        data=AR120p26s50by40;
        label=gnd;
    case lower('ar20')
        load("E:\ML\database\AR120p20s50by40_wrong_dataset.mat")
        data=AR120p20s50by40;
        label=gnd;
    case lower('yale')
        load("E:\ML\database\Yale5040165.mat")
        data=Yale5040165;
        label=gnd;
    case lower('cmupie')
        load("E:\ML\database\Pose091632p24s32by32.mat")
        data=Pose09_32by32;
        label=gnd;
    case lower('feret')
        load("E:\ML\database\FERET74040.mat")
        data=FERET74040;
        label=gnd;
    case lower('binaryalpha')
        load("E:\ML\database\binaryalphadigs_my.mat")
        data=binaryalphadigs_my;
        label=gnd;
    case lower('Breast_Cancer')
        load("E:\ML\database\Breast_Cancer.mat")
        data=Breast_Cancer;
        label=gnd;
        if isfield(options,'pca')
            options=rmfield(options,'pca');
        end
    case lower('Breast_Cancer_Coimbra')
        load("E:\ML\database\Breast_Cancer_Coimbra.mat")
        data=Breast_Cancer_Coimbra;
        label=gnd;
        if isfield(options,'pca')
            options=rmfield(options,'pca');
        end
    case lower('Breast_Cancer_Wisconsin_Diagnostic')
        load("E:\ML\database\Breast_Cancer_Wisconsin_Diagnostic.mat")
        data=Breast_Cancer_Diagnostic;
        label=gnd;
        if isfield(options,'pca')
            options=rmfield(options,'pca');
        end
    case lower('Breast_Cancer_Wisconsin_Original')
        load("E:\ML\database\Breast_Cancer_Wisconsin_Original.mat")
        data=Breast_Cancer_Original;
        label=gnd;
        if isfield(options,'pca')
            options=rmfield(options,'pca');
        end
    case lower('Breast_Cancer_Wisconsin_Prognostic')
        load("E:\ML\database\Breast_Cancer_Wisconsin_Prognostic.mat")
        data=Breast_Cancer_Prognostic;
        label=gnd;
        if isfield(options,'pca')
            options=rmfield(options,'pca');
        end
    case lower('Breast_Tissue')
        load("E:\ML\database\Breast_Tissue.mat")
        data=Breast_Tissue;
        label=gnd;
        if isfield(options,'pca')
            options=rmfield(options,'pca');
        end
    case lower('Mammography50x25size2class')
        load("E:\ML\database\Mammography50x25size2class.mat")
        data=Mammography;
        label=gnd;
    case lower('Mammography50x25size3class')
        load("E:\ML\database\Mammography50x25size3class.mat")
        data=Mammography;
        label=gnd;
    case lower('Mammography100x50size2class')
        load("E:\ML\database\Mammography100x50size2class.mat")
        data=Mammography;
        label=gnd;
    case lower('Mammography100x50size3class')
        load("E:\ML\database\Mammography100x50size3class.mat")
        data=Mammography;
        label=gnd;
end
end
%% Mnist Data include train set, test set and valid set
mnist_data={'Breast_Mnist','Pneumonia_Mnist'};
if sum(ismember(lower(mnist_data),lower(name)))
    if strcmpi('Breast_Mnist',name)
        load("E:\ML\database\Breast_Mnist.mat")
    elseif strcmpi('Pneumonia_Mnist',name)
        load("E:\ML\database\Pneumonia_Mnist.mat")
    end
    Xtrain=reshape(train,[size(train,1)*size(train,2) size(train,3)]);
    Xtest=reshape(test,[size(test,1)*size(test,2) size(test,3)]);
    Xvalid=reshape(valid,[size(valid,1)*size(valid,2) size(valid,3)]);
    tralabel=train_label+1;
    teslabel=test_label+1;
    vallabel=valid_label+1;
    % data processing
    if isfield(options,'normalize')
        switch lower(options.normalize)
            case {lower('FaceData01')}
                Xtrain=Xtrain/255;
                Xtest=Xtest/255;
            case {lower('Unit')}
                for i=1:size(Xtrain,2)
                    Xtrain(:,i)=Xtrain(:,i)/norm(Xtrain(:,i));
                end
                for i=1:size(Xtest,2)
                    Xtest(:,i)=Xtest(:,i)/norm(Xtest(:,i));
                end
            case {lower('zscore')}
                Xtrain=zscore(Xtrain');
                Xtrain=Xtrain';
                Xtest=zscore(Xtest');
                Xtest=Xtest';
            otherwise
                error('Normalize does not exist!')
        end
    end
    if isfield(options,'pca')
        pcaprojec=PCA(Xtrain,tralabel,options.pca);
        Xtrain=pcaprojec'*Xtrain;
        Xtest=pcaprojec'*Xtest;
        Xvalid=pcaprojec'*Xvalid;
    end
    X.train=Xtrain;
    X.test=Xtest;
    X.valid=Xvalid;
    X.tralabel=tralabel;
    X.teslabel=teslabel;
    X.vallabel=vallabel;
    return
end
%% 可以自己分训练集和测试集的数据集
Xtrain=[];
Xtest=[];
tralabel=[];
teslabel=[];

c_list=unique(label);
c=length(c_list);

dim=size(data);
if length(dim)==3
    for i=1:c
        index=find(label==i);
        ni=length(index);
        tranum=floor(ni*proportion);
        tesnum=ni-tranum;
        rand_list=randperm(ni);
        traindex=index(rand_list(1:tranum));
        tesindex=index(rand_list((tranum+1):end));
        Xtrain=cat(3,Xtrain,data(:,:,traindex));
        Xtest=cat(3,Xtest,data(:,:,tesindex));
        tralabel=[tralabel;i*ones(tranum,1)];
        teslabel=[teslabel;i*ones(tesnum,1)];
    end
    Xtrain=reshape(Xtrain,[size(Xtrain,1)*size(Xtrain,2) size(Xtrain,3)]);
    Xtest=reshape(Xtest,[size(Xtest,1)*size(Xtest,2) size(Xtest,3)]);
elseif length(dim)==2
    if length(label)==dim(1)%如果数据横放,调整回列放
        data=data';
    end
    for i=1:c
        index=find(label==i);
        ni=length(index);
        tranum=floor(ni*proportion);
        tesnum=ni-tranum;

        rand_list=randperm(ni);
        traindex=index(rand_list(1:tranum));
        tesindex=index(rand_list((tranum+1):end));
        Xtrain=[Xtrain data(:,traindex)];
        Xtest=[Xtest data(:,tesindex)];
        tralabel=[tralabel;i*ones(tranum,1)];
        teslabel=[teslabel;i*ones(tesnum,1)];
    end
end

Xtrain=double(Xtrain);
Xtest=double(Xtest);

% data processing
if isfield(options,'normalize')
    switch lower(options.normalize)
        case {lower('FaceData01')}
            Xtrain=Xtrain/255;
            Xtest=Xtest/255;
        case {lower('Unit')}
            for i=1:size(Xtrain,2)
                Xtrain(:,i)=Xtrain(:,i)/norm(Xtrain(:,i));
            end
            for i=1:size(Xtest,2)
                Xtest(:,i)=Xtest(:,i)/norm(Xtest(:,i));
            end
        case {lower('zscore')}
            Xtrain=zscore(Xtrain');
            Xtrain=Xtrain';
            Xtest=zscore(Xtest');
            Xtest=Xtest';
        otherwise
            error('Normalize does not exist!')
    end
end

% PCA
if isfield(options,'pca')
    pcaprojec=PCA(Xtrain,tralabel,options.pca);
    Xtrain=pcaprojec'*Xtrain;
    Xtest=pcaprojec'*Xtest;
end
X.train=Xtrain;
X.test=Xtest;
X.tralabel=tralabel;
X.teslabel=teslabel;

end

%=====================================================

function projection_matrix=PCA(data_set,tralabel,k)
%输入的是样本矩阵展开成的列向量再合并而成的矩阵
if size(data_set,1)==size(data_set,2)
    warning('Plesae ensure the data matrix is a column-vector matrix.')
end
if size(data_set,2)~=length(tralabel)
    data_set=data_set';
end

n=size(data_set,2);
mean_vector=mean(data_set,2);
% mean_vector=sum(data_set,2)/n;
scatter_matrix=(data_set-mean_vector)*(data_set-mean_vector)';
[eigenVectors,Y]=eig(scatter_matrix);
Y=double(Y);
[~,ind]=sort(diag(Y),"descend");
eigenVectors=eigenVectors(:,ind);
projection_matrix=eigenVectors(:,1:k);
end
