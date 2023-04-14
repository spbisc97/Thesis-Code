function Tester()
%% Test file for the functions
close all

%Choose Simulation
simulations=["Dyn","AttDyn","Cheaser","EulerCheaser"];
test=simulations(2);

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
    tspan=linspace(1,Hours*3600,Hours*3600);
    y0=[1;0;0;0;0;0;0];
    u=[0.1;0;0];
    [t,y]=ode45(@(t,y) Sat_Attitude_Dyn(t,y,u),tspan,y0);
    plot(t, quat2eul(y(:,1:4)))
    hold on
    plot(t,y(:,5:7))
    legend
end

%% Test The Cheaser
if test=="Cheaser"
    tspan=linspace(1,Hours*3600);
    y0=[1;-3;1;0;0;0];
    y_goal=[0;0;0;0;0;0];
    [t_traj,y_traj]=ode45(@(t,y) Cheaser(t,y,y_goal),tspan,y0);
    plot(t_traj,y_traj)
end

if test=="EulerCheaser"
    step=0.01;
    tspan=[1:step:Hours*3600]; %#ok
    y_traj=tspan.*zeros(6,1);
    u_traj=tspan.*zeros(3,1);
    y0=[1;-3;1;0;0;0];
    y_goal=[0;0;0;0;0;0];
    y=y0;
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
    plot(tspan,y_traj)
    figure()
    plot(tspan,u_traj)


end


