function wprox=proxl1(w,lambda)
wprox=sign(w).*max(abs(w)-lambda,0);
