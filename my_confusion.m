function [CM, acc] = my_confusion(Ctrues, Cpreds)

%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
CM=zeros(26,26);
A=Ctrues-Cpreds;
B=find(A);   %find index of nonzeros in A. those index indicate confusion element
N=size(Ctrues,1);
%for each confusion element, update record in CM
for i=1:size(B,1)
   CM(Ctrues(B(i)),Cpreds(B(i)))=CM(Ctrues(B(i)),Cpreds(B(i)))+1;
end
acc= (N-size(B,1))/N;
end

