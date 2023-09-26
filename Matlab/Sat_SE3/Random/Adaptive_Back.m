syms x_1 x_2 x_3
syms x_4 x_5 x_6
syms k4 k5 k6
syms k1 k2 k3
%syms rt omega mu


lambda1= k1+2;
lambda2= k2+1/2;
lambda3= k3+17/4;

mu=3.986004418*10^14;
rt=6.6*1e6;
omega=sqrt(mu/rt^3);

e=0;

h=rt^2+omega;

k=mu/h^(3/2);

abar1 = 1/2+lambda1^4/2 + lambda3^2+lambda1;
abar2 = lambda2^4/2 +lambda2;
abar3 = 1/2+lambda3^4/2 + lambda1^2+1/(1-e)^2 +lambda3;


a1 = k^4*(1+e)^3*(k4+abar1);
a2 = k^4*(1+e)^3*(k5+abar2);
a3 = k^4*(1+e)^3*(k6+abar3);


%u = -diag([a1 a2 a3])*[lambda1*x_1+x_4 lambda2*x_2+x_5 lambda3*x_3+x_6]';
u = -diag([a3 a1 a2])*[lambda3*x_3+x_6 lambda1*x_1+x_4 lambda2*x_2+x_5 ]';


k_coeff=[400 400 400 10 1 1 ]*10;

u_func=simplify(subs(u,[k1 k2 k3 k4 k5 k6],k_coeff));

syms x1 x2 x3 x4 x5 x6
u_func=subs(u_func,[x_1 x_2 x_3 x_4 x_5 x_6],[x2 x3 -x1 x5 x6 -x4 ]);

disp("\n u_func \n")

fun = matlabFunction(u_func,"File","backst",'Vars',{[x1 ;x2 ;x3 ;x4; x5; x6] });

disp(fun)