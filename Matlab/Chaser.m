function [t, y, u] = Chaser(t, y0, y_goal_traj, tspan) %#codegen
    %disp(t)
    coder.extrinsic('exist');
    
    if ~exist('y0', 'var')
        Return_lqr_values()
        return;
    end
    
    y0 = y0(:);
    [t, y, u] = forode(y_goal_traj, tspan, y0);
    
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
    options = odeset('AbsTol', 1e-5);
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
        [ode_t, ode_y] = ode45( ...
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
