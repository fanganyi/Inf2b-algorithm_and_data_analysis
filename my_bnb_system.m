clc;
clear;

%load data set
S =load ('data.mat');
Xtrn= S.dataset.train.images/255;
Ctrn=S.dataset.train.labels;
Xtst=S.dataset.test.images/255;
Ctrues=S.dataset.test.labels;
tic;
Cpreds=my_bnb_classify(Xtrn, Ctrn, Xtst, 1);
toc


[CM, acc] = my_confusion(Ctrues, Cpreds);
save('cm.mat','CM');
[n,a]=size(Xtst);
N=n
Nerrs=sum(sum(CM))
acc=acc
