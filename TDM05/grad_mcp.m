function [g] = grad_mcp( X,y,beta,lambda,gam )
I0 = find(abs(beta)<sqrt(eps));
I2 = find(abs(beta)>lambda*gam);
I1 = 1:length(beta);
I1(sort([I0 ; I2])) = [];
Gls = X'*(X*beta-y);
g(I2) = Gls(I2);
g(I1) = Gls(I1) + lambda*sign(beta(I1)) - beta(I1)/gam;
g(I0) = (abs(Gls(I0) - beta(I0)/gam) - lambda).*(abs(Gls(I0) - beta(I0)/gam) > lambda);
end