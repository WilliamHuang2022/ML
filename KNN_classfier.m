function accuracy=KNN_classfier(train_set,test_set,train_number,test_number,k)
accurate_number=0;
for i=1:size(test_set,2)
    distance_list=zeros(1,size(train_set,2));
    for j=1:size(train_set,2)
        dist=norm(train_set(:,j)-test_set(:,i));
        distance_list(j)=dist;
    end
    [~,ind]=sort(distance_list);
    site_list=ind(1:k);
    kind_list_of_train=ceil(site_list/train_number);
    kind_of_train=mode(kind_list_of_train);
    kind_of_test=ceil(i/test_number);
    if(kind_of_train==kind_of_test)
        accurate_number=accurate_number+1;
    end
end
accuracy=accurate_number/size(test_set,2);