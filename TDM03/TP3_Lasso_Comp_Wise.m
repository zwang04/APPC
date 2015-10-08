% cd /Users/stephane/Desktop/Canu/Enseign/APPC/TP/TP1_Lasso
% scanu@insa-rouen.fr
% septembre 2014

clear
prepare_housing

na = n/2;
randn('seed',10);
rand('seed',10);

q = 5;                  % adding useless variables
X = [X randn(n,q)];
X = (X - ones(n,1)*mean(X))./(ones(n,1)*std(X,1));

ind = randperm(n);
Xi = X(ind(1:na),:);
yi = y(ind(1:na));

Xt = X(ind(na+1:end),:);
yt = y(ind(na+1:end));

Xi = X/sqrt(n);
yi = y;
[na,p] = size(Xi);

 %%
 
 beta = zeros(p,1);
 nbitemax = 1000;
 Err = [];
 Errt = [];
 lambda = 25;   % for the LAsso  % q=5
 lambda = 45;   % for SCAD  % q=5
 lambda = 40;   % for MCP  % q=5
  gam = 5; % for MCP
[beta_MCP, lambdaM, df, ind_nextVar] = monMCP(Xi, yi, gam,0) ;
 for kk = 2:8
lambda =   lambdaM(kk);   % for MCP  % q=0
 
 %method = 'LASSO';
 %method = 'SCAD ';
 method = 'MCP  ';
 
for ii = 1:nbitemax
    ind = randperm(p);
    for jj = 1:p
        grad = - (Xi(:,ind(jj))'*(Xi*beta-yi - Xi(:,ind(jj))*beta(ind(jj))));
% M C P        
        if method == 'MCP  '
%                 gradC = (grad - lambda*sign(grad)*max(0,1 -abs( grad)/(lambda*gam)))/ (Xi(:,ind(jj))'*Xi(:,ind(jj)));
%                 beta(ind(jj)) = (grad > 0)*max(0,gradC) + (grad < 0)*min(0,gradC);
                if abs(grad) < lambda
                    beta(ind(jj)) = 0;
                elseif abs(grad) > gam*lambda
                    beta(ind(jj)) = grad;
                else
                    beta(ind(jj)) = sign(grad)*(abs(grad) - lambda)/(1-1/gam);
                end
% LA S S O
        elseif method == 'LASSO'
            gradC = abs(grad)-lambda;
            beta(ind(jj)) = sign(grad)*max(0, gradC) / (Xi(:,ind(jj))'*Xi(:,ind(jj)));
% S C A D            
        elseif method == 'SCAD '
            if abs(grad) > 3*lambda
                beta(ind(jj)) = grad / (Xi(:,ind(jj))'*Xi(:,ind(jj)));% the least square solution
            elseif abs(grad) > 3*lambda
                gradC = 2*abs(grad)-3*lambda;
                beta(ind(jj)) = sign(grad)*max(0, gradC) / (Xi(:,ind(jj))'*Xi(:,ind(jj)));
            else
                gradC = abs(grad)-lambda;
                beta(ind(jj)) = sign(grad)*max(0, gradC) / (Xi(:,ind(jj))'*Xi(:,ind(jj)));
            end
            
        end
    end
    Err = [Err norm(yi-Xi*beta)];
    Errt = [Errt norm(yt-Xt*beta)];
end
 
 Err;
 
 %% verifions le resultat pour le lasso
 H = [Xi'*Xi -Xi'*Xi;-Xi'*Xi Xi'*Xi];
 c = [Xi'*yi ; -Xi'*yi];
 A = ones(2*p,1);
 l = 10^-12;
 verbose = 0;
 
 b = sum(abs(beta));
     
 if method == 'LASSO'

     [xnew, dual_var, pos] = monqp(H,c,A,b,inf,l,verbose);
     
     Bpm = zeros(2*p,1);
     Bpm(pos) = xnew;
     
     [beta Bpm(1:p)-Bpm(p+1:end)]
     [ sum(abs(beta)) sum(abs(Bpm(1:p)-Bpm(p+1:end)))];
     
     
     %    plot(abs(Xi\yi),abs(beta),'o')
     
     %% verifions le resultat pour le MCP par DC
 elseif method == 'MCP  '
     
     % retieving the right b
     
      AA =  max(0,lambda - abs(beta_MCP(:,kk))/gam);
      b = AA'*abs(beta_MCP(:,kk));
           
     % DC as adaptive lasso
     Bpm = zeros(p,1);
     B = [];
     for ii = 1:50
         AA =  max(0,lambda - abs(Bpm)/gam);
         A = [AA ; AA];
         [xnew, dual_var, pos] = monqp(H,c,A,b,inf,l,verbose);
         Bpm = zeros(2*p,1);
         Bpm(pos) = xnew;
         Bpm = Bpm(1:p)-Bpm(p+1:end);
         
%         B = [B Bpm];
         
%          tic
%          cvx_begin
%             cvx_precision best
%             variables beta_cvx(p)
%             dual variable d
%             minimize(  norm(yi - Xi*beta_cvx,2)  )
%             subject to
%                 d  : A(1:p)'*abs(beta_cvx) <= b;
%         cvx_end
%          toc
    end
     
      
     
 end
 
 
 [beta Bpm beta_MCP(:,kk)]
 [ sum(abs(beta)) sum(abs(Bpm))];

 end
 
 %%
%  
%  function [f,fprim] = scad(x,l,a)
%     if ~exist('a')
%         a=4;
%     end
%     f = zeros(size(x));
%     fprim=f;
%     for i=1:length(x)
%         if x(i)>=a*l            
%             f(i)=x(i);
%             fprim(i)=1;
%         else
%             if x(i)>2*l
%                 f(i)=((a-1)*x(i)-a*l)/(a-2);
%                 fprim(i)=(a-1)/(a-2);     
%             else
%                 f(i)=max(x(i)-l,0);
%                 fprim(i)=x(i)>l;
%             end
%         end        
%     end
% end
