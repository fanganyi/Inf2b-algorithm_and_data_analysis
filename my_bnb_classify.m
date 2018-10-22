function [Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
M = size(Xtrn,1);
N = size(Xtst,1);
X=binarisation(Xtrn,threshold);
T=binarisation(Xtst,threshold);
T1=1-T;
p=zeros(N,26);

for i=1:26
    index= find(Ctrn==i);
    Nk=size(index,1);        %Nk represent the total number of trains in class i
    values=X(index,:);        %extract all the trns of class i, VALUES IS A Nk by D matrix
    prob=(sum(values))/Nk;  %P(bi|Ck), a 1 by D matrix
   %p(tst | CLASS=i)  
    p(:,i)=prod(((bsxfun(@power,prob,T)).*(bsxfun(@power,1-prob,T1))),2)*Nk/M;
    
end
[~,in]=max(p,[],2);
Cpreds=in;

end

