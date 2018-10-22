clc;
clear;

%load data set
S =load ('data.mat');
Xtrn= double(S.dataset.train.images)/255;
Ctrn=S.dataset.train.labels;
Xtst=double(S.dataset.test.images)/255;
Ctrues=S.dataset.test.labels;

epsilon=0.01;
tic;
[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon);
toc

[CM, acc] = my_confusion(Ctrues, Cpreds);
save('cm.mat','CM');
save('m26.mat','Ms');
save('cov26.mat','Covs')
n=size(Xtst,1);
N=n
Nerrs=sum(sum(CM))
acc=acc