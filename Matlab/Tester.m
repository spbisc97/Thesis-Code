function Tester()
    %% Test file for the functions
    close all

    %Choose Simulation
    simulations=["Dyn","AttDyn","Cheaser","EulerCheaser"];
    test=simulations(4);

    % Simulation Time
    %Days=0.0001;
    Hours=0.01;%24*Days;


    %% Test The Satellites Dynamics

    if test=="Dyn"
        tspan=linspace(1,Hours*3600);
        y0=[0;0;1;0;0;0];
        u=[0;0;0];
        [t,y]=ode45(@(t,y) Sat_Translational_Dyn(t,y,u),tspan,y0);
        plot(t,y)
    end


    if test=="AttDyn"
        tspan=1:0.5:Hours*3600;
        %remember the rule for the initial rotation
        eulZYX=[0,0,0];
        q0=eul2quat(eulZYX)';
        y0=[q0;0.1;0;0];
        u=[0;0;0];
        [t,y]=ode45(@(t,y) Sat_Attitude_Dyn(t,y,u),tspan,y0);
        nexttile
        plot(t, quat2eul(y(:,1:4),"ZYX"))
        legend
        nexttile
        plot(t, vecnorm(y(:,1:4)')')
        legend
        nexttile
        plot(t,y(:,5:7))
        legend
    end

    %% Test The Cheaser
    if test=="Cheaser"
        tspan=linspace(1,Hours*3600);
        eulZYX=[0,0,0];
        q0=eul2quat(eulZYX)';
        y0_att=[q0;0;0;0];
        y0_tra=[1;-3;1;0;0;0];

        q_goal=eul2quat([0,0,0])';
        y_goal_tra=[0;0;0;0;0;0];
        y_goal_att=[q_goal;0;0;0];

        [t_traj,y_traj]=ode45(@(t,y) Cheaser(t,y,[y_goal_tra;y_goal_att]),tspan,[y0_tra;y0_att]);
        plot(t_traj,y_traj)
    end

    if test=="EulerCheaser"
        step=0.01;
        tspan=[1:step:Hours*3600]; %#ok
        y_traj=tspan.*zeros(13,1);
        u_traj=tspan.*zeros(6,1);

        eulZYX=[0,0,0];
        q0=eul2quat(eulZYX)';
        y0_att=[q0;0;0;0];
        y0_tra=[1;-3;1;0;0;0];

        q_goal=eul2quat([pi/2,0,0])';
        y_goal_att=[q_goal;0;0;0];
        y_goal_tra=[0;0;0;0;0;0];

        y=[y0_tra;y0_att];
        y_goal=[y_goal_tra;y_goal_att];
        y_traj(:,1)=y;
        counter=1;
        for t=tspan

            [dy,u]=Cheaser(t,y,y_goal);
            y = y + step*dy;

            y_traj(:,counter)=y;
            u_traj(:,counter)=u;

            counter=counter+1;
            %t = t + step;
        end
        figure()
        nexttile

        plot(tspan,y_traj(1:3,:))
        label = {'X_d','Y_d','Z_d'};
        yline(y_goal(1:3),'-',label)
        legend('X','Y','Z');
        title("Lv Lh Position")


        nexttile
        plot(tspan,y_traj(4:6,:))
        label = ["X'_d","Y'_d","Z'_d"];
        yline(y_goal(4:6),'-',label)
        legend("X'","Y'","Z'");
        title("Lv Lh Velocity")

        nexttile

        nexttile
        plot(tspan,quat2eul(y_traj(7:10,:)',"ZYX")')
        label = ["X_d","Y_d","Z_d"];
        yline(quat2eul(y_goal(7:10)'),'-',label)
        legend("X","Y","Z");
        title("Euler attitude")


        nexttile
        plot(tspan,y_traj(11:13,:))
        label = ["w_x_d","w_y_d","w_z_d"];
        yline(y_goal(11:13),'-',label)
        legend("w_x","w_y'","w_z'");
        title("Angular velocity")

        nexttile
        plot(tspan,vecnorm(y_traj(7:10,:))')
        legend("Norm");
        title("Quaternion Norm")

        figure()
        nexttile
        plot(tspan,u_traj(1:3,:))
        legend("X","Y","Z");
        title("u_traj - translation")


        nexttile
        plot(tspan,u_traj(4:6,:))
        legend("X","Y","Z");
        title("u_traj - attitude")






    end


