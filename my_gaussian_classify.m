function [Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
D= size(Xtrn,2);
N = size(Xtst,1);
Ms=zeros(D,26);
Covs=zeros(D,D,26);
likehood=zeros(N,26);
for k=1:26
    
    %get mean
    index = find(Ctrn==k);%find all the trains in class k in binary form
    Nk=size(index,1);        %Nk represent the total number of trains in class k
    values=(Xtrn(index,:));   %extact values in class k
    Ms(:,k) =((mean((values),1))); %compute mean
 
    %get voc
     sub=values-((Ms(:,k))');
     Covs(:,:,k)=(((sub')*sub)/(Nk-1))+eye(D)*epsilon;
     
     %get log likehood
     likehood(:,k)= -0.5*sum((Xtst-((Ms(:,k))'))/(Covs(:,:,k)).*(Xtst-((Ms(:,k))')),2)-sum(log(diag(chol(Covs(:,:,k)))))-D*log(2*pi)/2;
     
end
    [~,in]= max(likehood,[],2);  %find MLE
    Cpreds = in;                 %index of second dimension of MLE is predicted class
end

