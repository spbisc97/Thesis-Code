function Tester(test_n)
    %% Test file for the functions
    close all
    addpath('Auxiliary');
    tic
    %Choose Simulation
    simulations=["Dyn","AttDyn","Chaser","EulerChaser_Traj","EulerChaser_Point"];
    if ~exist('test_n','var') || ~isnumeric(test_n)
        test_n=5;
    end
    test=simulations(test_n);

    % Simulation Time
    % Days=1;
    Hours=12;


    %% Test The Satellites Dynamics

    if test=="Dyn"
        tspan=1:0.1:Hours*3600;
        x0=[0;1800;0];
        v0=[1;0;0];
        M0=10;
        u=@(t) 0.01*[sin(100*t),0,0];
        y0=[x0;v0;M0];
        [t,y]=ode45(@(t,y) Sat_Translational_Dyn(t,y,u(0)),tspan,y0);
        toc
        translation_plotter(t,y,[0;0;0;0;0;0]'.*ones(length(t),1));
    end


    if test=="AttDyn"
        tspan=1:0.5:Hours*3600;
        %remember the rule for the initial rotation
        eulZYX=[0,0,0];
        q0=eul2quat(eulZYX)';
        va0=[0;0;0];
        M0=10;

        y0=[q0;va0;M0];

        u=@(t) 0.1*[sin(t);sin(t);sin(t)];
        [t,y]=ode45(@(t,y) Sat_Attitude_Dyn(t,y,u(t)),tspan,y0);
        toc
        attitude_plotter(t,y,[1;0;0;0;0;0;0]'.*ones(length(t),1));

    end

    %% Test The Chaser
    if test=="Chaser"
        step=10;
        tspan=1:step:Hours*3600;
        eulZYX=[1,0,0];
        q0=eul2quat(eulZYX)';
        y0_att=[q0;0;0;0];
        y0_tra=[1;0;0;0;0;0];
        y0_mass=10;

        q_goal=eul2quat([0,0,0])';
        y_goal_att=[q_goal;0;0;0];
        y_goal_tra=[0;0;0;0;0;0];
        y_goal_mass=y0_mass;

        y0=[y0_tra;y0_att;y0_mass];
        y_goal=[y_goal_tra;y_goal_att;y_goal_mass];

        [t_traj,y_traj]=ode45(@(t,y) Chaser(t,y,y_goal),tspan,y0);
        toc
        
        %reconstruct output from ode45
        u_traj=t_traj.*zeros(1,6);
        for i=1:length(t_traj)
            [~,u]=Chaser(t_traj(i),y_traj(i,:),y_goal);
            u_traj(i,:)=u;
        end
        plotter(t_traj,y_traj,y_goal'.*ones(length(t_traj),1),u_traj);
    end

    if test=="EulerChaser"
        step=0.1;
        tspan=[1:step:Hours*3600]; %#ok
        y_traj=tspan.*zeros(14,1);
        y_goal_traj=tspan.*zeros(14,1);
        u_traj=tspan.*zeros(6,1);
        

       
%% Goal Trajectory
        q_goal=eul2quat([0,0,0])';
        y_goal_att=[q_goal;0;0;0];
        y_goal_tra=[0;1800;0;1;0;0];
        y_goal_mass=Sat_params().fuel_mass;
        y_goal=[y_goal_tra;y_goal_att;y_goal_mass];
        y_goal_traj(:,1)=y_goal;

%% Initial Conditions

        eulZYX=[0,0,0];
        q0=eul2quat(eulZYX)';
        y0_att=[q0;0;0;0];
        y0_tra=[0;1800;0;1;0;0];
        y0_mass=Sat_params().fuel_mass;
        y=[y0_tra;y0_att;y0_mass];
        y_traj(:,1)=y;
        counter=1;

        for t=tspan

            f_goal_traj=@(y) [(y(2)/2)/1000;-(y(1)*2)/1000];
            y_goal_traj(4:5,counter)=f_goal_traj(y_goal_traj(1:2,counter));
            y_goal_traj(1:2,counter)=y_goal_traj(1:2,counter)+y_goal_traj(4:5,counter)*step;
        
            counter=counter+1;
            y_goal_traj(:,counter)=y_goal_traj(:,counter-1);

            %t = t + step;
        end

       
        
        counter=1;

        for t=tspan
           
            [dy,u]=Chaser(t,y,y_goal_traj(:,counter));
            y = y + step*dy;

            y_traj(:,counter)=y;
            u_traj(:,counter)=u;
            

            counter=counter+1;
            %t = t + step;
        end
        toc
        plotter(tspan',y_traj',y_goal_traj(:,1:length(tspan))',u_traj')


    end
    if test=="EulerChaser_Traj"
                step=10;
        tspan=1:step:Hours*3600;
        eulZYX=[1,0,0];
        q0=eul2quat(eulZYX)';
        y0_att=[q0;0;0;0];
        y0_tra=[1;0;0;0;0;0];
        y0_mass=10;

        q_goal=eul2quat([0,0,0])';
        y_goal_att=[q_goal;0;0;0];
        y_goal_tra=[0;0;0;0;0;0];
        y_goal_mass=y0_mass;

        y0=[y0_tra;y0_att;y0_mass];
        y_goal=[y_goal_tra;y_goal_att;y_goal_mass];

        [t_traj,y_traj]=ode45(@(t,y) Chaser(t,y,y_goal),tspan,y0);
        toc
        
        %reconstruct output from ode45
        u_traj=t_traj.*zeros(1,6);
        for i=1:length(t_traj)
            [~,u]=Chaser(t_traj(i),y_traj(i,:),y_goal);
            u_traj(i,:)=u;
        end
        plotter(t_traj,y_traj,y_goal'.*ones(length(t_traj),1),u_traj);
    end

    if test=="EulerChaser_Point"
        step=0.1;
        tspan=[1:step:Hours*3600]; %#ok
        y_traj=tspan.*zeros(14,1);
        y_goal_traj=tspan.*zeros(14,1);
        u_traj=tspan.*zeros(6,1);
        

       
%% Goal Trajectory
        q_goal=eul2quat([0.2,0,0])';
        y_goal_att=[q_goal;0;0;0];
        y_goal_tra=[0;0;0;0;0;0];
        y_goal_mass=Sat_params().fuel_mass;
        y_goal=[y_goal_tra;y_goal_att;y_goal_mass];
        y_goal_traj(:,1)=y_goal;
        f_goal_traj=@(y) [0;0];

%% Initial Conditions

        eulZYX=[0,0,0];
        q0=eul2quat(eulZYX)';
        y0_att=[q0;0;0;0];
        y0_tra=[0;0;0;0;0;0];
        y0_mass=Sat_params().fuel_mass;
        y=[y0_tra;y0_att;y0_mass];
        y_traj(:,1)=y;
        counter=1;

        for t=tspan
        
            counter=counter+1;
            y_goal_traj(:,counter)=y_goal_traj(:,counter-1);

            %t = t + step;
        end

       
        
        counter=1;

        for t=tspan
           
            [dy,u]=Chaser(t,y,y_goal_traj(:,counter));
            y = y + step*dy;

            y_traj(:,counter)=y;
            u_traj(:,counter)=u;
            

            counter=counter+1;
            %t = t + step;
        end
        toc
        plotter(tspan',y_traj',y_goal_traj(:,1:length(tspan))',u_traj')

        
    end
end


    %% Plotter
    function plotter(t,y,y_goal,u)
        %we will use vertical vectors
        

        translation_plotter(t,y(:,1:6),y_goal(:,1:6))  ;      

        euler_y=[quat2eul(y(:,7:10),"ZYX"),y(:,11:13)];
        euler_y_goal=[quat2eul(y_goal(:,7:10),"ZYX"),y_goal(:,11:13)];
        attitude_plotter(t,euler_y,euler_y_goal) ;      

        fig=figure();
        fig.Name="Norm&Norm";

        nexttile
        plot(t,vecnorm(y(:,7:10)')')
        legend("Norm");
        title("Quaternion Norm");

        nexttile
        plot(t,y(:,14));
        legend("Mass");
        title("Vehicle Mass");

        if exist('u','var');
            u_plotter(t,u);
        end
    end
    
    function fig=u_plotter(t,u)
        fig=figure();
        fig.Name="Control";
        nexttile;
        plot(t,u(:,1:3));
        legend("X","Y","Z");
        title("u_traj - translation");


        nexttile;
        plot(t,u(:,4:6));
        legend("X","Y","Z");
        title("u_traj - attitude");
    end

    function fig=translation_plotter(tspan,y_traj,y_goal_traj)
        %plot Translation
        fig=figure();
        fig.Name="Translation";
        tiledlayout(2, 3)
        
        nexttile
        plot(tspan,y_traj(:,1))
        hold on
        plot(tspan,y_goal_traj(:,1),'--')
        legend('X','X_d');
        title("Lv Lh Position")
        nexttile
        plot(tspan,y_traj(:,2))
        hold on
        plot(tspan,y_goal_traj(:,2),'--')
        legend('Y','Y_d');
        title("Lv Lh Position")
        nexttile
        plot(tspan,y_traj(:,3))
        hold on
        plot(tspan,y_goal_traj(:,3),'--')
        legend('Z','Z_d');
        title("Lv Lh Position")


        nexttile
        plot(tspan,y_traj(:,4))
        hold on
        plot(tspan,y_goal_traj(:,4),'--')
        legend("X'","X'_d");
        title("Lv Lh Velocity")

        nexttile
        plot(tspan,y_traj(:,5))
        hold on
        plot(tspan,y_goal_traj(:,5),'--')
        legend("Y'","Y'_d");
        title("Lv Lh Velocity")

        nexttile
        plot(tspan,y_traj(:,6))
        hold on
        plot(tspan,y_goal_traj(:,6),'--')
        legend("Z'","Z'_d");
        title("Lv Lh Velocity")

    end


    function fig=attitude_plotter(tspan,y,y_goal)
        %plot attitude with euler
        fig=figure();
        fig.Name="Attitude";
        nexttile
        plot(tspan,y(:,1))
        hold on
        plot(tspan,y_goal(:,1)','--');
        legend("X","X_d");
        title("Euler attitude")
        nexttile
        plot(tspan,y(:,2))
        hold on
        plot(tspan,y_goal(:,2)','--');
        legend("Y","Y_d");
        title("Euler attitude")
        nexttile
        plot(tspan,y(:,3))
        hold on
        plot(tspan,y_goal(:,3)','--');
        legend("Z","Z_d");
        title("Euler attitude")


        nexttile
        plot(tspan,y(:,4))
        hold on
        plot(tspan,y_goal(:,4),'--')
        legend("w_x","w_x_d");
        title("Angular velocity")
        nexttile

        plot(tspan,y(:,5))
        hold on
        plot(tspan,y_goal(:,5),'--')
        legend("w_y'","w_y_d");
        title("Angular velocity")

        nexttile
        plot(tspan,y(:,6))
        hold on
        plot(tspan,y_goal(:,6),'--')
        legend("w_z'","w_z_d");
        title("Angular velocity")
    
    end