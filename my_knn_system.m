clc;
clear;

%load data set
S =load ('data.mat');
Xtrn= single(S.dataset.train.images)/255;
Ctrn=S.dataset.train.labels;
Xtst=single(S.dataset.test.images)/255;
Ctrues=S.dataset.test.labels;
Ks=[1; 3; 5; 10; 20];
tic;
Cpreds=my_knn_classify(Xtrn, Ctrn, Xtst, Ks);
toc

%k=1
[CM1, acc1] = my_confusion(Ctrues, Cpreds(:, 1));
save('cm1.mat','CM1');

%k=3
[CM3, acc3] = my_confusion(Ctrues, Cpreds(:, 2));
save('cm3.mat','CM3');

%k=5
[CM5, acc5] = my_confusion(Ctrues, Cpreds(:,3));
save('cm5.mat','CM5');

%k=10
[CM10, acc10] = my_confusion(Ctrues, Cpreds(:,4));
save('cm10.mat','CM10');

%k=20
[CM20, acc20] = my_confusion(Ctrues, Cpreds(:,5));
save('cm20.mat','CM20');
[n,a]=size(Xtst);
k=Ks;
N=[n;n;n;n;n];
acc=[acc1;acc3;acc5;acc10;acc20];
Nerrs=[sum(sum(CM1));sum(sum(CM3));sum(sum(CM5));sum(sum(CM10));sum(sum(CM20))];
table(k,N,Nerrs,acc)