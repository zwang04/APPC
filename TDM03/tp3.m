%% tp3

%% Scalable Lasso
clear all
close all
clc

%% A. le cas moindre carré 

% 1, le pb
    %sig = 0.1;
    sig = 0.5;
    
    n = 1000;
    p = 1000;
    Bvrai = ones(p,1);
    X = rand (n,p);
    y = X* Bvrai + sig*randn(n,1);

    % out of memory
    % B_mc=(X'*X)\(X'*y)

% 2, la Solution
    X=(X-ones(n,1)*mean(X));
    Beta = zeros(p,1); % initialisation de b
    %nbiteMax=10;
    nbiteMax=100;
    nbiteMax=20;
    for i=1:nbiteMax  % tant qu'on n'a pas convergé
        ind=randperm(p);
        for k=1:p
            z= y-X*Beta + X(:,ind(k))*Beta(ind(k));
            Beta(ind(k))=(X(:,ind(k))'*z)/(X(:,ind(k))'*X(:,ind(k)));
        end
        norm(X*Beta-y); % pour vérifier la cout 
        errA(i)=(X*Beta-y)'*(X*Beta-y);
        norm(Beta-Bvrai);
    end
    BetaA = Beta;
    
    
%% B

% 1, le pb
    %sig = 0.1;
    sig = 1;
    
    n = 1000;
    p = 800;
    Bvrai = ones(p,1);
    X = rand (n,p);
    y = X* Bvrai + sig*randn(n,1);

    % out of memory
    % B_mc=(X'*X)\(X'*y)

% 2, la Solution

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
   
%% C

% 1, le pb
    %sig = 0.1;
    sig = 1;
    
    n = 1000;
    p = 800;
    Bvrai = ones(p,1);
    X = rand (n,p);
    y = X* Bvrai + sig*randn(n,1);

    % out of memory
    % B_mc=(X'*X)\(X'*y)

% 2, la Solution

    X=(X-ones(n,1)*mean(X));
    Beta = zeros(p,1); % initialisation de b
    %nbiteMax=10;
    nbiteMax=100;
    nbiteMax=75;
    LAM = .1 ;
    for i=1:nbiteMax  % tant qu'on n'a pas convergé
        ind=randperm(p);
        for k=1:p
            z= y-X*Beta + X(:,ind(k))*Beta(ind(k));
            Beta(ind(k))=(X(:,ind(k))'*z)/(X(:,ind(k))'*X(:,ind(k)));
            
            %d
            Beta(ind(k))=sign(Beta(ind(k)))*max(0,abs(Beta(k))-LAM);

        end
        norm(X*Beta-y); % pour vérifier la cout 
        errC(i)=(X*Beta-y)'*(X*Beta-y);
        norm(Beta-Bvrai);
    end
    
    BetaC = Beta ;