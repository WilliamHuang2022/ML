function accuracy=KNN_Classfier(traResult,tesResult,tralabel,teslabel,k)
% tralabel,teslabel为训练集和测试集标签向量
if size(traResult,1)==size(traResult,2)||size(tesResult,1)==size(tesResult,2)
    warning('Plesae ensure the data matrix is a column-vector matrix.')
end
if size(traResult,2)~=length(tralabel)&&size(tesResult,2)~=length(teslabel)
    traResult=traResult';
    tesResult=tesResult';
end

accurate_number=0;
for i=1:length(teslabel)
    distance_list=zeros(1,length(tralabel));
    for j=1:length(tralabel)
        distance_list(j)=norm(traResult(:,j)-tesResult(:,i));
    end
    [~,ind]=sort(distance_list);
    site_list=ind(1:k);
    kind=mode(tralabel(site_list));
    if kind==teslabel(i)
        accurate_number=accurate_number+1;
    end
end
accuracy=accurate_number/length(teslabel);
end