%% tp4

close all
clear all
clc


%%
sig = 0.5;
    
n = 1000;
p = 1000;
Bvrai = ones(p,1);
X = rand (n,p);
y = X* Bvrai + sig*randn(n,1);

X=(X-ones(n,1)*mean(X));
Beta = zeros(p,1); % initialisation de b
%nbiteMax=10;
nbiteMax=100;
nbiteMax=75;
LAM = .5 ;
for i=1:nbiteMax  % tant qu'on n'a pas convergé
    ind=randperm(p);
    for k=1:p
        z= y-X*Beta + X(:,ind(k))*Beta(ind(k));
        Beta(ind(k))=(X(:,ind(k))'*z)/(X(:,ind(k))'*X(:,ind(k)));
        
        %c
        Beta(ind(k))=Beta(ind(k))/((1+LAM/(X(:,ind(k))'*X(:,ind(k)))));

    end
    norm(X*Beta-y); % pour vérifier la cout 
    errB(i)=(X*Beta-y)'*(X*Beta-y);
    norm(Beta-Bvrai);
end
    
    BetaB = Beta ;

%%
d = p;
w = Beta;
x = X;
%lambda = 10;
lambda = 0.5;
for i =1:100
    for k = 1:d
        wtemp = w;
        wtemp(k) = 0;
        sk = y - x*wtemp;
        w(k) = sign(x(:,k)'*sk)*max(0,abs(x(:,k)'*sk)-lambda)/((x(:,k))'*(x(:,k)));


        correl = -x'*(y-x*w);
        epsi = 1.e-6;
        indNonZero = find(abs(w)>=epsi);
        indNonZero = find(abs(w)<epsi);

        if abs(correl+lambda*sign(w(k)))<epsi & abs(correl)<=lambda
          break;
        else
           toOptimise = [indZero(find(abs(correl(indZero))>=lambda)),indNonZero(find(abs(abs(correl(indZero))-lambda))>=epsi))
        end
    end
end

% group=1:d;
% group=reshape(group',2,d/2)';
% nbgroup=size(group,1);
% for k=1:nbgroup
% ind=group(k,:);
% [X(:,ind),~]=qr(X(:,ind),0);
% end;