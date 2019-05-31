function xnew=minmaxnorm(x,a,b)
%min-max normalization of x to interval [a,b].
%Inputs: 
%x: data
%[a,b]: interval where to scale i.e. [0,1] unit interval.
%Output:
%xnew: scaled data
[n,m]=size(x);
amin=min(x);
amax=max(x);
diffv=amax-amin;
bb=b*ones(n,m);
aa=a*ones(n,m);
xnew=(x-repmat(amin,n,1))./repmat(diffv,n,1).*(bb-aa)+aa;