function y = norms( varargin )

%NORMS   Computation of multiple vector norms.
%   NORMS( X ) provides a means to compute the norms of multiple vectors
%   packed into a matrix or N-D array. This is useful for performing
%   max-of-norms or sum-of-norms calculations.
%
%   All of the vector norms, including the false "-inf" norm, supported
%   by NORM() have been implemented in the NORMS() command.
%     NORMS(X,P)           = sum(abs(X).^P).^(1/P)
%     NORMS(X)             = NORMS(X,2).
%     NORMS(X,inf)         = max(abs(X)).
%     NORMS(X,-inf)        = min(abs(X)).
%   If X is a vector, these computations are completely identical to
%   their NORM equivalents. If X is a matrix, a row vector is returned
%   of the norms of each column of X. If X is an N-D matrix, the norms
%   are computed along the first non-singleton dimension.
%
%   NORMS( X, [], DIM ) or NORMS( X, 2, DIM ) computes Euclidean norms
%   along the dimension DIM. NORMS( X, P, DIM ) computes its norms
%   along the dimension DIM.
%
%   Disciplined convex programming information:
%       NORMS is convex, except when P<1, so an error will result if these
%       non-convex "norms" are used within CVX expressions. NORMS is
%       nonmonotonic, so its input must be affine.

persistent P
if isempty( P ),
    P.map = cvx_remap( { 'constant' ; 'l_convex' ; ...
        { 'p_convex', 'n_concave', 'affine' } } );
    P.funcs = { @norms_1, @norms_1, @norms_2 };
    P.zero = 0;
    P.reduce = true;
    P.reverse = false;
    P.constant = 1;
    P.fname = 'norms';
    P.dimarg = 3;
end
[ sx, x, p, dim ] = cvx_get_dimension( varargin, 3 );
if nargin < 2 || isempty(p),
    p = 2;
elseif ~( isnumeric(p) && numel(p)==1 && isreal(p) && p >= 1 ),
    cvx_throw( 'Second argument must be a scalar between 1 and +Inf, inclusive.' );
end
if sx(dim) == 0,
    sx(dim) = 1;
    y = zeros( sx, 1 );
    if isa( x, 'cvx' ), y = cvx( y ); end
elseif sx(dim) == 1,
    y = abs( x );
elseif p == 1,
    y = sum( abs( x ), dim );
elseif p == Inf,
    y = max( abs( x ), [], dim );
else
    y = cvx_reduce_op( P, x, p, dim );
end

function y = norms_1( x, p )
y = sum( abs( x ) .^ p, 1 ) .^ ( 1 / p );

function y = norms_2( x, p ) %#ok
[nx,nv] = size(x);
if p == 2,
    cvx_begin
        epigraph variable y( 1, nv ) nonnegative_
        { cvx_linearize(x), y } == lorentz( [ nx, nv ], 1, ~isreal( x ) ); %#ok
    cvx_end
else
    cvx_begin
        variable z( nx, nv )
        epigraph variable y( 1, nv ) nonnegative_
        if isreal(x), cmode = 'abs'; else cmode = 'cabs'; end
        { cat( 3, z, repmat(y,[nx,1]) ), cvx_linearize(x) } ...
            == geo_mean_cone( [nx,nv,2], 3, [1/p,1-1/p], cmode ); %#ok
        sum( z ) == y; %#ok
    cvx_end
end

% Copyright 2005-2014 CVX Research, Inc.
% See the file LICENSE.txt for full copyright information.
% The command 'cvx_where' will show where this file is located.
