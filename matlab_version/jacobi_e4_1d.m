function e = jacobi_ijk_1d(p, alpha, beta)
%
% jacobi_ijk_1d.m - Evaluate the inner product of 1d Jacobi-chaos triplets
%
% Syntax     e = jacobi_ijk_1d(p, alpha, beta)
%
% Input:     p = order of Jacobi-chaos
%            alpha, beta = parameters of Jacobi-chaos (alpha, beta>-1)
% Output:    e = (p+1)x(p+1)x(p+1) matrix containing the result.
% 
% NO WARNING MESSAGE IS GIVEN WHEN PAPAMETERS ARE OUT OF RANGE.
%
% By Dongbin Xiu   03/25/2002
%

tolerance=1e-10;
e = zeros(p+1,p+1,p+1,p+1);

np = ceil((4*p+1)/2);

[z,w] = zwgj(np, alpha, beta);  

factor = 2^(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+2);

J = zeros(p+1, np);
for order=0:p
  J(order+1, :) = jacobf(z', order, alpha, beta);
end
  
for i=1:p+1
  for j=i:p+1
     for k=j:p+1
        for l=k:p+1
	  e(i,j,k,l) = sum(J(i,:).*J(j,:).*J(k,:).*J(l,:).*w')/factor;
          if abs(e(i,j,k,l)-round(e(i,j,k,l))) < tolerance
             e(i,j,k,l) = round(e(i,j,k,l));
          end

	   e(i,k,j,l) = e(i,j,k,l);
	   e(j,i,k,l) = e(i,j,k,l);
	   e(j,k,i,l) = e(i,j,k,l);
	   e(k,i,j,l) = e(i,j,k,l);
	   e(k,j,i,l) = e(i,j,k,l);

	   e(i,k,l,j) = e(i,j,k,l);
	   e(j,i,l,k) = e(i,j,k,l);
	   e(j,k,l,i) = e(i,j,k,l);
	   e(k,i,l,j) = e(i,j,k,l);
	   e(k,j,l,i) = e(i,j,k,l);

	   e(i,l,k,j) = e(i,j,k,l);
	   e(j,l,i,k) = e(i,j,k,l);
	   e(j,l,k,i) = e(i,j,k,l);
	   e(k,l,i,j) = e(i,j,k,l);
	   e(k,l,j,i) = e(i,j,k,l);

	   e(l,i,k,j) = e(i,j,k,l);
	   e(l,j,i,k) = e(i,j,k,l);
	   e(l,j,k,i) = e(i,j,k,l);
	   e(l,k,i,j) = e(i,j,k,l);
	   e(l,k,j,i) = e(i,j,k,l);
        end
      end
   end
end



