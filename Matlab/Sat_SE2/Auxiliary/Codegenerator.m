p=Sat_params();

y0=10*1e3;

initialconditions=[0 y0 0 y0/2000 0 0]';
days=1;
timespan=1:0.05:3600*24*days;


codegen Sat_dyn_Lin -args {0,initialconditions,[1.1,3.2]',p}