function S=simi(x,y,p)

S=(1-abs(x.^p-y.^p)).^(1/p);
