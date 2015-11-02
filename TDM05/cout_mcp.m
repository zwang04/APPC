function [cout ] = cout_mcp( Xi,yi,beta,lambda,gam )
cout = .5*(Xi*beta-yi)'*(Xi*beta-yi) + sum( (abs(beta)>gam*lambda)*gam*lambda^2/2 + (abs(beta)<=gam*lambda).*(lambda*abs(beta) - beta.^2/(2*gam)) );
end