train_set=AR120p20s50by40_train;
test_set=AR120p20s50by40_test;
c=120;                                           %训练集类数
train_withinnumber=7;                          %训练集类内个数
test_withinnumber=7;                             %测试集类内个数

train_vectors=matrix2vectors(train_set);
test_vectors=matrix2vectors(test_set);

projection_matrix=PCA(train_vectors,50);          %PCA数据处理

train_vectors=projection_matrix'*train_vectors;
test_vectors=projection_matrix'*test_vectors;

train_matrixs=zeros(size(train_vectors,1),train_withinnumber,c);
for i=1:c
    matrix=train_vectors(:,1+(i-1)*train_withinnumber:i*train_withinnumber);
    train_matrixs(:,:,i)=matrix;
end
accurate_number=0;                                      %正确个数
panduanchisu=0;
for m=1:size(test_vectors,2)
    y=test_vectors(:,m);
    distance_list=[];
    
    for n=1:c
        X=train_matrixs(:,:,n);
        beta=rand(train_withinnumber,1)/3;                       %β初始化
        a=100;                                                %正则项系数
        b=0.02;                                                   %初始步长
        f=(norm(X*beta-y))^2+a*norm(beta,1);
        times=40;                                             %迭代次数
        runtime=25;                                             %一个参数下降次数
        %x_list=[0:1:times];
        %y_list=[f];
        for i=1:times
            for j=1:train_withinnumber
                for k=1:runtime
                    h=k;
                    beta(j)=beta(j)+b;
                    if (norm(X*beta-y))^2+a*norm(beta,1)>f
                        b=-b;
                        beta(j)=beta(j)+b;
                    end
                    f=(norm(X*beta-y))^2+norm(beta,1);
                end
            end
            f=(norm(X*beta-y))^2+norm(beta,1);
            %y_list=[y_list f];
        end
        distance=(norm(X*beta-y))^2;
        distance_list=[distance_list distance];
    end
    site=find(distance_list==min(distance_list));
    if ceil(m/test_withinnumber)==site
        accurate_number=accurate_number+1;
    end
    panduanchisu=panduanchisu+1
end
accuracy=accurate_number/size(test_set,3)
    
        