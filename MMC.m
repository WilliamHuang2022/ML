function W=MMC(X,tranum,p)
[d,n]=size(X);

mean_t=mean(X,2);
meanX=zeros(d,n/tranum);
for i=1:n/tranum
    meanX(:,i)=mean(X(:,((i-1)*tranum+1):i*tranum),2);
end
Sb=zeros(d,d);
for i=1:n/tranum
    Sb=Sb+(meanX(:,i)-mean_t)*(meanX(:,i)-mean_t)';
end
Sw=zeros(d,d);
for i=1:n/tranum
    for j=1:tranum
        Sw=Sw+(X(:,(i-1)*tranum+j)-meanX(:,i))*(X(:,(i-1)*tranum+j)-meanX(:,i))';
    end
end

[V,D]=eig(Sb-Sw);
[~,ind]=sort(diag(D),'descend');
W=V(:,ind(1:p));

end