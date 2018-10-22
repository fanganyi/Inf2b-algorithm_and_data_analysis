function [Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst)
% use K means classify to improve gaussian classify

D= size(Xtrn,2);

%get 26 random rows, each indicate a cluster center
ran=rand(26,D);
r=zeros(1,D);
%classify Xtst to the nearst cluster center
C=my_knn_classify(ran, (linspace(1,26,26))', Xtst, linspace(1,1,1));

% recompute each cluster mean
for i=1:26
    while r~=ran(i,:)
        r=ran(i,:);
        ran(i,:) = mean(vertcat((Xtst(C==i,:)),ran(i,:)),1);
    end
end
%use gaussian to find the most likely predicted class of each cluster
[A, ~, ~] = my_gaussian_classify(Xtrn, Ctrn, ran, 0.01);
% compute Xtst predicted class
for i=1:26
    C(find(C==i))=A(i);
    Cpreds=C;
end
    