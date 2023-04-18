function [loss]=L21norm(M)
n=size(M,1);
loss=0;
for i=1:n
    loss=loss+norm(M(i,:));
end
end