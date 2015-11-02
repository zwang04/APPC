function w=ProximalSparseRegression(X,y,lambda,epsi)

[n,d] = size(X);
w=zeros(d,1);
stepsize = 1/norm(X'*X);
grad= -X'*(y-X*w);
seuil = stepsize*grad;

while seuil < epsi
    grad= -X'*(y-X*w);
    w=w - stepsize*grad;
    w=proxl1(w,stepsize*lambda);
end

end