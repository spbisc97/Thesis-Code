function [tspout, yout] = RK4(dydt,tsp,y0)
arguments
    dydt function_handle
    tsp (:,1)
    y0 (:,1)
end
% RK4  Classical 4th order Runge-Kutta solver.
%   y = RK4(t,y,tsp,args) uses 4th order Runge-Kutta to
%   integrate the system of differential equations y' = F(t,y)
%   from time t to time tsp, with initial conditions y.
%   The time step is chosen automatically.
%   The function F(t,y) must return a column vector.
%   Each row in the solution array y corresponds to a time
%   returned in the column vector t.
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
    k_1 = dydt(tsp(i),y(:,i));
    k_2 = dydt(tsp(i)+0.5*h,y(:,i)+0.5*h*k_1);
    k_3 = dydt((tsp(i)+0.5*h),(y(:,i)+0.5*h*k_2));
    k_4 = dydt((tsp(i)+h),(y(:,i)+k_3*h));
    y(:,i+1) = y(:,i) + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    % y(:,i+1)=RK4_Step(dydt,tsp(i),y(:,i),h);

end
tspout=tsp;
yout=y.';
end