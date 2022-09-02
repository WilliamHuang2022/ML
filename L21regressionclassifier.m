function accuracy=L21regressionclassifier(testdata,W,withinnumber)
%输入三维测试集
accuratenumber=0;
n=size(testdata,3);
testvectors=matrix2vectors(testdata);
testvectors1=[testvectors;ones(1,n)];
testY=testvectors1'*W;
for i=1:n
    testy=testY(i,:);
    site=find(testy==max(testy));
    if site==ceil(i/withinnumber)
        accuratenumber=accuratenumber+1;
    end
end
accuracy=accuratenumber/n;
