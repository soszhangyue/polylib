function jchaos = Hermite_expand_nd(x, coef, norder)
%
% hermite_expand.m - Evaluates the Hermite-chaos expansion at given data points
%
% Syntax:   jchaos = hermite_expand_nd(x, coef, order)
% 
% Input :   x  =  (npt,ndim) data points in matrix form where expansion to be evaluated
%           coef = array with length (P+1) containning expansion coeffcients
%                  of P-th order Jacobi-chaos
%           order = specified order for the expansion to be evaluated; 
%                   if order is not in the range defined by coef (1,P), i.e.,
%                   order>P or <=0, then the full order P from coef will be
%                   used. (Thus, 'order' provides a way to evaluate
%                   the chaos at expansion order lower than P.)
%
% Output:   jchaos = values of expansion at x, stored in the same format as x
%
% NO WARNING MESSAGE IS GIVEN WHEN PAPAMETERS ARE OUT OF RANGE.
%
% Code generated by Dongbin Xiu 09/03/2012.
%

npt = length(x(:,1));
ndim = length(x(1,:));
nterm = min(nchoosek(ndim+norder,ndim), length(coef));

%P = length(coef)-1;
%if ((order <= 0) | (order > P))
%  order = P;
%end

jchaos = zeros(npt,1);

for m = 1:npt
   poly = HermiteF_nd(x(m,:), ndim, norder);
   for n = 1:nterm
      jchaos(m) = jchaos(m) + coef(n)*poly(n);
	end
end
