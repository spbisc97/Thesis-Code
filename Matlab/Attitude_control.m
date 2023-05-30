function [u,e]=Attitude_control(t,y,y_goal_traj,tspan) %#codegen
    %u=[0;0;0];
    e=[0;0;0;0;0;0;0];
    
    y_goal=interp1(tspan,y_goal_traj,t).';
    %disp([y,y_goal])
    u=quat_err_rate(y,y_goal);
end
%% Attitude Helping Functions
function [u,e]=quat_err_rate(y,y_goal)
    q_e=quat_tracking_error(y(1:4),y_goal(1:4));
    e=[q_e;0;0;0];
    parameters=Sat_params();
    p=-q_e(2:4);%/(1.4-norm(q_e(2:4))^2);
    d=y_goal(5:7)-y(5:7);
    % kp=1 * 1e-4;
    % kd=1000 * 1e-4;
    kp=(q_e(1))*(1e-4)*parameters.invI;% energy shaping optimizations %Mattioni 2022
    kd=2e-2;
    
    u=kp*p+kd*d;
end

function mat=omega(q)
    % mat for crossproduct btw quaternions
    % q(1) is the scalar part
    
    mat=[q(1),-q(2),-q(3),-q(4);...
        q(2),q(1),q(4),-q(3);...
        q(3),-q(4),q(1),q(2);...
        q(4),q(3),-q(2),q(1)];
end

function e=quat_tracking_error(q,q_d)
    % overkill using quatinv for the error
    q_d_inv=quatinv(q_d')';
    e=omega(q_d_inv)*q;
    e=e(:);
    if norm(e)>1.1
        disp("error norm error")
        disp(norm(e))
    end
end





