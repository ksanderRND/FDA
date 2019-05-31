function c=entropyDeLuca2(x)
xtmp=x(x<1);
xnew=xtmp(xtmp>0);
c=-sum(xnew.*log(xnew)+(1-xnew).*log(1-xnew));

