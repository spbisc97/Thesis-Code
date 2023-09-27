function [tspout, yout] = Euler(dydt,tsp,y0)
arguments
    dydt function_handle
    tsp (:,1)
    y0 (:,1)
end
% RK4  Classical 4th order Runge-Kutta solver.

%F_xy=Sat_dyn;
h = tsp(2) - tsp(1);
steps=size(tsp,1);
% h is the time step
% ensure that y0 is a column vector
y = [zeros(size(y0,1),steps)];
% preallocate y
y(:,1)=y0;

for i=1:(steps-1)
    % calculation loop
    y(:,i+1) = y(:,i) + dydt(tsp(i),y(:,i))*h;  % main equation

end
tspout=tsp;
yout=y.';
end