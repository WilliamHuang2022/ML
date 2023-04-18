function accuracy=LRC(train_set,test_set,lamuda,c,train_number,test_number)
%train_set=AR120p20s50by40_train;         输入三维数据
%test_set=AR120p20s50by40_test;
%lamuda=0.1;                             %正则项系数
%c=120;                                    %数据集类数
%train_number=7;                         %训练集类内个数
%test_number=7;                           %测试集类内个数
%---------------------------------------------------
n1=size(train_set,1);
n2=size(train_set,2);
accurate_number=0;
train_vectors=matrix2vectors(train_set);
test_vectors=matrix2vectors(test_set);
%将训练集按一类一层列向量矩阵布置
train_matrixs=zeros(n1*n2,train_number,c);
for i=1:c
    matrix=train_vectors(:,1+(i-1)*train_number:i*train_number);
    train_matrixs(:,:,i)=matrix;
end

%—————————————————————————————
for i=1:size(test_vectors,2)
    y=test_vectors(:,i);
    distance_list=[];
    for j=1:c
        Xj=train_matrixs(:,:,j);
        beta=((Xj'*Xj+lamuda*eye(train_number)))^(-1)*Xj'*y;
        yj=Xj*beta;
        distance=norm(y-yj);
        distance_list=[distance_list distance];
    end
    minimum=min(distance_list);
    site=find(distance_list==minimum);
    if ceil(i/test_number)==site
        accurate_number=accurate_number+1;
    end
end
accuracy=accurate_number/size(test_set,3);
    
    
    
