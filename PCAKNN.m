function accuracy=PCAKNN(projected_vectors_of_train_set,projected_vectors_of_test_set,k)
accurate_number=0;
for i=1:size(projected_vectors_of_test_set,2)
    distance_list=[];
    for j=1:size(projected_vectors_of_train_set,2)
        dist=norm(projected_vectors_of_train_set(:,j)-projected_vectors_of_test_set(:,i));
        distance_list=[distance_list dist];
    end
    arranged_distance_list=sort(distance_list);
    chosed_distance_list=arranged_distance_list(1:k);
    kind_list_of_train=[];
    for m=1:k
        site=find(distance_list==chosed_distance_list(m));
        kind=ceil(site/7);                                                 %修改训练类的数
        kind_list_of_train=[kind_list_of_train kind];
    end
    kind_of_train=mode(kind_list_of_train);
    kind_of_test=ceil(i/3);                                                %修改测试类的数
    if(kind_of_train==kind_of_test)
        accurate_number=accurate_number+1;
    end
end
accuracy=accurate_number/size(projected_vectors_of_test_set,2);           