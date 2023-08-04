function Tester(test_n, length_hours)
    %% Test file for the functions
    close all
    addpath('Auxiliary');
    tic
    %Choose Simulation
    simulations = ["TranDyn", "AttDyn", "Chaser_traj", "Chaser_Point_Attitude", "Chaser_Point_Trajectory"];
    
    if ~exist('test_n', 'var') || ~isnumeric(test_n)
        test_n = 4;
    end
    
    if ~exist('length_hours', 'var') || ~isnumeric(length_hours)
        length_hours = 0.5;
    end
    
    % Codegen if you want
    % codegen Sat_Attitude_Dyn -args {1,[0.5 0.5 0.5 0.5 0.3 0.2 -0.1 10],[0.001  0.02  -0.4]}
    % codegen Sat_Translational_Dyn -args {1,[10 -10 10 0.3 0.3 0.8 10],[0.001  0.02  -0.4]}
    % codegen Sat_dyn -args {1,[10 10 -10 0.3 0.3 0.8 0.5 0.5 0.5 0.5 0.3 0.2 -0.1 10],[0.001  0.02  -0.4],[0.001  0.02  -0.4]}
    %codegen('Sat_dyn', ...
    % '-args',...
    % {1, [10; 10; -10; 0.3; 0.3; 0.8; 0.5; 0.5; 0.5; 0.5; 0.3; 0.2; -0.1; 10], [0.001; 0.02; -0.4], [0.001; 0.02; -0.4]},...
    % '-o', 'Sat_dyn');
    
    test = simulations(test_n);
    
    % Simulation Time
    % Days=1;
    Hours = length_hours;
    
    text = "starting "+test + "for "+Hours + " hours";
    
    disp(text)
    
    %% Only Test Tranjational Dyn
    if test == "TranDyn"
        tspan = 1:0.1:Hours * 3600;
        x0 = [0; 1800; 0];
        v0 = [1; 0; 0];
        M0 = 10;
        u = @(t) 0.01 * [sin(100 * t), 0, 0];
        y0 = [x0; v0; M0];
        [t, y] = ode45(@(t, y) Sat_Translational_Dyn(t, y, u(0)), tspan, y0);
        toc
        translation_plotter(t, y, [0; 0; 0; 0; 0; 0]' .* ones(length(t), 1));
    end
    
    %% Only Test Attitude Dyn
    if test == "AttDyn"
        tspan = 1:0.5:Hours * 3600;
        %remember the rule for the initial rotation
        eulZYX = [0, 0, 0];
        q0 = eul2quat(eulZYX)';
        va0 = [0; 0; 0];
        M0 = 10;
        
        y0 = [q0; va0; M0];
        
        u = @(t)1e-15 * [sin(t / 5e+5); 1e-2 * cos(t / 1e+4); 1e-3 * sin(t / 1e+3)];
        [t, y] = ode45(@(t, y) Sat_Attitude_Dyn(t, y, u(t)), tspan, y0);
        toc
        attitude_plotter(t, y, [1; 0; 0; 0; 0; 0; 0]' .* ones(length(t), 1));
        
    end
    
    if test == "Chaser_traj"
        step = 0.5;
        tspan = 1:step:Hours * 3600;
        y_traj = tspan' .* zeros(1, 14);
        y_goal_traj = tspan' .* zeros(1, 14);
        u_traj = tspan' .* zeros(1, 6);
        
        %% Goal Trajectory
        q_goal = eul2quat([0, 0, 0])';
        y_goal_att = [q_goal; 0; 0; 0];
        y_goal_tra = [0; 1800; 0; 1; 0; 0];
        y_goal_mass = Sat_params().fuel_mass;
        y_goal = [y_goal_tra; y_goal_att; y_goal_mass];
        y_goal_traj(1, :) = y_goal';
        
        %% Initial Conditions
        
        eulZYX = [0, 0, 0];
        q0 = eul2quat(eulZYX)';
        y0_att = [q0; 0; 0; 0];
        y0_tra = [0; 1800; 0; 1; 0; 0];
        y0_mass = Sat_params().fuel_mass;
        y = [y0_tra; y0_att; y0_mass];
        y0 = y;
        y_traj(1, :) = y;
        counter = 1;
        f_goal_traj = @(y) [(y(2) / 2) / 1000; - (y(1) * 2) / 1000];
        
        for t = tspan(2:end)
            counter = counter + 1;
            y_goal_traj(counter, :) = y_goal_traj(counter - 1, :);
        end
        
        [~, k_y] = ode45(@(t, y) f_goal_traj(y), tspan, y_goal_traj(1, 1:2));
        y_goal_traj(:, 1:2) = k_y;
        counter = 1;
        
        for t = tspan(2:end)
            counter = counter + 1;
            y_goal_traj(counter, 4:5) = f_goal_traj(y_goal_traj(counter, 1:2));
        end
        
        [t_traj, y_traj, u_traj] = Chaser(t, y0, y_goal_traj, tspan);
        toc
        plotter(tspan, y_traj, y_goal_traj, u_traj)
        
    end
    
    %% Chase Point Attitude
    if test == "Chaser_Point_Attitude"
        step = 0.5;
        tspan = [1:step:Hours * 3600]; %#ok
        y_traj = tspan' .* zeros(1, 14);
        y_goal_traj = tspan' .* zeros(1, 14);
        u_traj = tspan' .* zeros(1, 6);
        
        %% Goal Trajectory
        q_goal = eul2quat([0, 0, 0])';
        y_goal_att = [q_goal; 0; 0; 0];
        y_goal_tra = [0; 0; 0; 0; 0; 0];
        y_goal_mass = Sat_params().fuel_mass;
        y_goal = [y_goal_tra; y_goal_att; y_goal_mass];
        y_goal_traj(1, :) = y_goal.';
        %f_goal_traj=@(y) [0;0];
        
        %% Initial Conditions
        eulZYX = (rand(1, 3) - 0.5) * pi;
        %eulZYX = [1, -0.5, 3];
        q0 = eul2quat(eulZYX)';
        y0_att = [q0; 0; 0; 0];
        y0_tra = [0; 0; 0; 0; 0; 0];
        y0_mass = Sat_params().fuel_mass;
        y0 = [y0_tra; y0_att; y0_mass];
        counter = 1;
        
        for t = tspan(1: end - 1)
            
            counter = counter + 1;
            y_goal_traj(counter, :) = y_goal_traj(counter - 1, :);
            
            %t = t + step;
        end
        
        [tspan, y_traj, u_traj] = Chaser(t, y0, y_goal_traj, tspan);
        
        toc
        plotter(tspan, y_traj, y_goal_traj, u_traj)
        
    end
    
    %% Chase Point Trajectory
    if test == "Chaser_Point_Trajectory"
        step = 0.5;
        tspan = [1:step:Hours * 3600]; %#ok
        y_traj = tspan' .* zeros(1, 14);
        y_goal_traj = tspan' .* zeros(1, 14);
        u_traj = tspan' .* zeros(1, 6);
        
        %% Goal Trajectory
        q_goal = eul2quat([0, 0, 0])';
        y_goal_att = [q_goal; 0; 0; 0];
        y_goal_tra = [0; 0; 0; 0; 0; 0];
        y_goal_mass = Sat_params().fuel_mass;
        y_goal = [y_goal_tra; y_goal_att; y_goal_mass];
        y_goal_traj(1, :) = y_goal.';
        %f_goal_traj=@(y) [0;0];
        
        %% Initial Conditions
        eulZYX = [0, -0, 0];
        q0 = eul2quat(eulZYX)';
        y0_att = [q0; 0; 0; 0];
        y0_tra = [1000; 400; 10; 0; 0; 0];
        y0_mass = Sat_params().fuel_mass;
        y0 = [y0_tra; y0_att; y0_mass];
        counter = 1;
        
        for t = tspan(1: end - 1)
            
            counter = counter + 1;
            y_goal_traj(counter, :) = y_goal_traj(counter - 1, :);
            
            %t = t + step;
        end
        
        [tspan, y_traj, u_traj] = Chaser(t, y0, y_goal_traj, tspan);
        
        toc
        plotter(tspan, y_traj, y_goal_traj, u_traj)
        
    end
    
end

%% Plotter
function plotter(t, y, y_goal, u)
    %we will use vertical vectors
    
    translation_plotter(t, y(:, 1:6), y_goal(:, 1:6));
    
    euler_y = [quat2eul(y(:, 7:10), "ZYX"), y(:, 11:13)];
    euler_y_goal = [quat2eul(y_goal(:, 7:10), "ZYX"), y_goal(:, 11:13)];
    attitude_plotter(t, euler_y, euler_y_goal);
    
    fig = figure();
    fig.Name = "Norm&Norm";
    
    nexttile
    plot(t, vecnorm(y(:, 7:10)')')
    legend("Norm");
    title("Quaternion Norm");
    
    nexttile
    plot(t, y(:, 14));
    legend("Mass");
    title("Vehicle Mass");
    
    if exist('u', 'var')
        u_plotter(t, u);
    end
    
end

function fig = u_plotter(t, u)
    fig = figure();
    fig.Name = "Control";
    nexttile;
    stairs(t, u(:, 1:3));
    legend("X", "Y", "Z");
    title("u_traj - translation");
    
    nexttile;
    
    stairs(t, u(:, 4:6));
    legend("X", "Y", "Z");
    title("u_traj - attitude");
end

function fig = translation_plotter(tspan, y_traj, y_goal_traj)
    %plot Translation
    fig = figure();
    fig.Name = "Translation";
    tiledlayout(2, 3)
    
    nexttile
    plot(tspan, y_traj(:, 1))
    hold on
    plot(tspan, y_goal_traj(:, 1), '--')
    legend('X', 'X_d');
    title("Lv Lh Position")
    nexttile
    plot(tspan, y_traj(:, 2))
    hold on
    plot(tspan, y_goal_traj(:, 2), '--')
    legend('Y', 'Y_d');
    title("Lv Lh Position")
    nexttile
    plot(tspan, y_traj(:, 3))
    hold on
    plot(tspan, y_goal_traj(:, 3), '--')
    legend('Z', 'Z_d');
    title("Lv Lh Position")
    
    nexttile
    plot(tspan, y_traj(:, 4))
    hold on
    plot(tspan, y_goal_traj(:, 4), '--')
    legend("X'", "X'_d");
    title("Lv Lh Velocity")
    
    nexttile
    plot(tspan, y_traj(:, 5))
    hold on
    plot(tspan, y_goal_traj(:, 5), '--')
    legend("Y'", "Y'_d");
    title("Lv Lh Velocity")
    
    nexttile
    plot(tspan, y_traj(:, 6))
    hold on
    plot(tspan, y_goal_traj(:, 6), '--')
    legend("Z'", "Z'_d");
    title("Lv Lh Velocity")
    
end

function fig = attitude_plotter(tspan, y, y_goal)
    %plot attitude with euler
    fig = figure();
    fig.Name = "Attitude";
    nexttile
    plot(tspan, y(:, 1))
    hold on
    plot(tspan, y_goal(:, 1)', '--');
    legend("X", "X_d");
    title("Euler attitude")
    nexttile
    plot(tspan, y(:, 2))
    hold on
    plot(tspan, y_goal(:, 2)', '--');
    legend("Y", "Y_d");
    title("Euler attitude")
    nexttile
    plot(tspan, y(:, 3))
    hold on
    plot(tspan, y_goal(:, 3)', '--');
    legend("Z", "Z_d");
    title("Euler attitude")
    
    nexttile
    plot(tspan, y(:, 4))
    hold on
    plot(tspan, y_goal(:, 4), '--')
    legend("w_x", "w_x_d");
    title("Angular velocity")
    nexttile
    
    plot(tspan, y(:, 5))
    hold on
    plot(tspan, y_goal(:, 5), '--')
    legend("w_y'", "w_y_d");
    title("Angular velocity")
    
    nexttile
    plot(tspan, y(:, 6))
    hold on
    plot(tspan, y_goal(:, 6), '--')
    legend("w_z'", "w_z_d");
    title("Angular velocity")
    
end
