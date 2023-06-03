function [t, y, u] = Chaser(t, y0, y_goal_traj, tspan) %#codegen
    
    y0 = y0(:);
    [t, y, u] = forrk4(y_goal_traj, tspan, y0);
    
end

function u = retrive_controls(y, y_goal_traj, tspan)
    disp("retrive_controls")
    u = zeros(size(y, 1), 6);
    
    for counter = 1:1:size(y, 1)
        u(counter, :) = [
            Translation_control(tspan(counter), y(counter, 1:6)', y_goal_traj(:, 1:6), tspan); ...
            Attitude_control(tspan(counter), y(counter, 7:13)', y_goal_traj(:, 7:13), tspan)]';
    end
    
end

function [t, y, u] = nonstiffode(y_goal_traj, tspan, y0)
    options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
    [t, y] = ode23( ...
        @(t, y) ...
        Sat_dyn(t, y, ...
        Translation_control(t, y(1:6), y_goal_traj(:, 1:6), tspan), ...
        Attitude_control(t, y(7:13), y_goal_traj(:, 7:13), tspan) ...
        ), ...
        tspan, ...
        y0, ...
        options);
    
    u = retrive_controls(y, y_goal_traj, tspan);
end

function [t, y, u] = forode(y_goal_traj, tspan, y0)
    options = odeset('AbsTol', 1e-3);
    y_ode = y0;
    len = length(tspan);
    t = tspan';
    y = zeros(len, 14);
    u = zeros(len, 6);
    %t(1) = tspan(1);
    y(1, :) = y0(:)';
    u_tranlational = Translation_control(t(1), y(1, 1:6).', y_goal_traj(:, 1:6), tspan); %later mass will have to be passed
    u_attitude = Attitude_control(t(1), y(1, 7:13).', y_goal_traj(:, 7:13), tspan); %later mass will have to be passed
    u(1, :) = [u_tranlational(:).', u_attitude(:).'];
    
    for counter = 1:1:len - 1
        
        ode_tsp = tspan(counter:counter + 1);
        [ode_t, ode_y] = ode23tb( ...
            @(t, y) Sat_dyn(t, y, u_tranlational, u_attitude), ...
            ode_tsp, ...
            y(counter, :)', ...
            options);
        t(counter + 1) = [ode_t(end)];
        y(counter + 1, :) = [ode_y(end, :)];
        u_tranlational = Translation_control(t(counter), y(counter, 1:6).', y_goal_traj(:, 1:6), tspan); %later mass will have to be passed
        u_attitude = Attitude_control(t(counter), y(counter, 7:13).', y_goal_traj(:, 7:13), tspan); %later mass will have to be passed
        u(counter + 1, :) = [u_tranlational(:).', u_attitude(:).'];
    end
    
end

function [t, y, u] = forrk4(y_goal_traj, tspan, y0)
    y_ode = y0;
    len = length(tspan);
    t = tspan';
    y = zeros(len, 14);
    u = zeros(len, 6);
    %t(1) = tspan(1);
    y(1, :) = y0(:)';
    for counter = 1:1:len - 1
        u_tranlational = Translation_control(t(counter), y(counter, 1:6).', y_goal_traj(:, 1:6), tspan); %later mass will have to be passed
        u_attitude = Attitude_control(t(counter), y(counter, 7:13).', y_goal_traj(:, 7:13), tspan); %later mass will have to be passed
        
        ode_tsp = [tspan(counter) tspan(counter + 1)];
        [ode_t, ode_y, ode_u] = rk4( ...
            y(counter, :)', ...
            ode_tsp, ...
            [u_tranlational(:).', u_attitude(:).'].');
        t(counter + 1) = [ode_t];
        y(counter + 1, :) = [ode_y];
        u(counter, :) = [ode_u];
    end
    
end


function [t, y, u] = rk4(y,tsp,args)
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
    % h is the time step
    
    y = y(:);                                           % ensure that y is a column vector
    y = [y zeros(length(y),length(tsp)-1)];
    % preallocate y
    
    for i=1:(length(tsp)-1)
        % calculation loop
        [k_1,u] = Sat_dyn(tsp(i),y(:,i),args(1:3),args(4:6));
        k_2 = Sat_dyn(tsp(i)+0.5*h,y(:,i)+0.5*h*k_1,args(1:3),args(4:6));
        k_3 = Sat_dyn((tsp(i)+0.5*h),(y(:,i)+0.5*h*k_2),args(1:3),args(4:6));
        k_4 = Sat_dyn((tsp(i)+h),(y(:,i)+k_3*h),args(1:3),args(4:6));
        y(:,i+1) = y(:,i) + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
    end
    y=y(:,end);
    t=tsp(end);
end