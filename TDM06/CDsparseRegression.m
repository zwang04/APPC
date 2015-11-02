function w=CDsparseRegression(X,y,lambda,epsi)


[n,d] = size(X);
w=zeros(d,1);
for i =1:100
    for k = 1:d
        wtemp = w;
        wtemp(k) = 0;
        sk = y - X*wtemp;
        w(k) = sign(X(:,k)'*sk)*max(0,abs(X(:,k)'*sk)-lambda)/((X(:,k))'*(X(:,k)));


        %correl = -X'*(y-X*w);
        %epsi = 1.e-6;
        %indZero = find(abs(w)>=epsi);
        %indNonZero = find(abs(w)<epsi);

%         if abs(correl+lambda*sign(w(k)))<epsi & abs(correl)<=lambda
%           break;
%         else
%            toOptimise = [indZero(find(abs(correl(indZero))>=lambda)),indNonZero(find(abs(abs(correl(indZero))-lambda))>=epsi)]
%         end
    end
end