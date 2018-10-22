function [Cpreds] = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = size(Xtst,1);
L= size(Ks,1);
 Cpreds = zeros(N,L); %INITIAL OUTPUT

xx=(sum(Xtst.^2,2));  %Xi*Xi
yy=(sum(Xtrn.^2,2))';  %Yj*Yj
dis=xx-2.*Xtst*(Xtrn')+yy; %Xi*Xi-2Xi*Yj+Yj*Yj

 [~,j] = sort(dis,2,'ascend');

 for k=1:L  %for each value of k
     a=j(:,1:Ks(k));  %find first k values in sorted dis
     modes=mode(Ctrn(a),2); %find the mode value, this mode value is the predicted class
     colum=modes(:,1);
     Cpreds(:,k)=colum;  %store predicted cladd in output
 
 end
 
end

