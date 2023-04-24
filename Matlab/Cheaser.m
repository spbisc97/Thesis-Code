function [dy,u] = Cheaser(t,y,y_goal) %#codegen
    %disp(t)
    if ~exist('y','var')
        Return_lqr_values()
        return;
    end

    y=y(:);


    

    u_tranlational=translation_control(y(1:6),y_goal(1:6));%later mass will have to be passed
    u_attitude=attitude_control(y(7:13),y_goal(7:13));%later mass will have to be passed

    
    [dy,u]=Sat_dyn(t,y,u_tranlational,u_attitude);
    %u=[u_tranlational;u_attitude];
end




%% Control Functions
function [u,e]=translation_control(y,y_goal)
    %u=0;e=0;
    e=y_goal(1:6)-y(1:6);
    u=lqr_control_tra(e);
end

function [u,e]=attitude_control(y,y_goal)
    %u=[0;0;0];
    e=[0;0;0;0;0;0;0];
    u=quat_err_rate(y,y_goal);

end

%% Translation Helping Functions
function u=lqr_control_tra(error)
    u=get_k_tra()*(error);
end

function K=get_k_tra()
    %K=lqr(get_A_tra(),get_B_tra(),get_Q_tra(),get_R_tra());
    %pause
    K=[...
        3.1623   -0.0000   -0.0000    3.1633    0.0000    0.0000;...
        0.0000    3.1623    0.0000   -0.0000    3.1633    0.0000;...
        0.0000   -0.0000    3.1623    0.0000   -0.0000    3.1633;];
end

function A=get_A_tra()
    n= 6.3135e-04;
    %Init Vars using     %F_LV_LH
    A=[ 0 0 0 1 0 0 ;...
        0 0 0 0 1 0;...
        0 0 0 0 0 1;...
        3*n^2 0 0 0 2*n 0;...
        0 0 0 -2*n 0 0;...
        0 0 -n^2 0 0 0;...
        ];
end
function B= get_B_tra()
    B=[ 0 0 0;...
        0 0 0;...
        0 0 0;...
        1 0 0;...
        0 1 0;...
        0 0 1;];
end
function Q=get_Q_tra()
    Q=diag([1,1,1,1,1,1]);
end
function R=get_R_tra()
    R=diag([1,1,1])*1e-7;
end
function Return_lqr_values()
    K=lqr(get_A_tra(),get_B_tra(),get_Q_tra(),get_R_tra());
    %string([K,[";...";";...";";];"]])
    NoInput="No Input has been provided, the K lqr values are returned ";
    disp(NoInput)
    disp(K)
end

%% Attitude Helping Functions
function [u,e]=quat_err_rate(y,y_goal)
    q_e=quat_tracking_error(y(1:4),y_goal(1:4));
    e=[q_e;0;0;0];
    u=-q_e(2:4)/(1-norm(q_e(2:4))^2)-y(5:7);
    u=u/100;
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
    q_d_inv=quatinv(q_d')';
    e=omega(q_d_inv)*q;
    e=e(:);
    if norm(e)>1.1
        disp("error norm error")
        disp(norm(e))
    end
end





