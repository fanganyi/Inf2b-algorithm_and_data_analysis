clc;
clear;

%load data set
S =load ('data.mat');
Xtrn= double(S.dataset.train.images)/255;
Ctrn=S.dataset.train.labels;
Xtst=double(S.dataset.test.images)/255;
Ctrues=S.dataset.test.labels;

tic;
[Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst);
toc

[CM, acc] = my_confusion(Ctrues, Cpreds);
save('cm improved.mat','CM');

n=size(Xtst,1);
N=n
Nerrs=sum(sum(CM))
acc=acc