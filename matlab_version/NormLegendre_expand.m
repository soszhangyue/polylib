function jchaos = NormLegendre_expand(x, coef, order)
%
% NormLegendre_expand.m  -  Evaluates the normalized Legendre-chaos expansion 
%                     at given data points
%
% Syntax:   jchaos = NormLegendre_expand(x, coef, order)
% 
% Input :   x  =  data points in matrix form where expansion to be evaluated
%           coef = array with length (P+1) containning expansion coeffcients
%                  of P-th order Jacobi-chaos
%           order = specified order for the expansion to be evaluated; 
%                   if order is not in the range defined by coef (1,P), i.e.,
%                   order>P or <=0, then the full order P from coef will be
%                   used. (in a word, 'order' provides a way to evaluate
%                   the chaos at expansion order lower than P.)
%
% Output:   jchaos = values of expansion at x, stored in the same format as x
%
% NO WARNING MESSAGE IS GIVEN WHEN PAPAMETERS ARE OUT OF RANGE.
%
% Code generated by Dongbin Xiu 06/13/2005.
%

P = length(coef)-1;
if ((order <= 0) | (order > P))
  order = P;
end

jchaos = zeros(size(x));

for i=1:(order+1)
jchaos = jchaos + coef(i)*LegendreF_Normal(x, i-1);
end