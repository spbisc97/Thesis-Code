function yout=RK4_Step(dydt,t,y,h)
arguments
    dydt function_handle
    t
    y
    h
end

% calculation loop
k_1 = dydt(t,y);
k_2 = dydt(t+0.5*h,y+0.5*h*k_1);
k_3 = dydt((t+0.5*h),(y+0.5*h*k_2));
k_4 = dydt((t+h),(y+k_3*h));
yout = y + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation

end