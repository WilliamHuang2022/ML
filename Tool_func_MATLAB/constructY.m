function [Y]=constructY(label)
n=length(label);
c=length(unique(label));
Y=zeros(n,c);
for i=1:n
    Y(i,label(i))=1;
end
end