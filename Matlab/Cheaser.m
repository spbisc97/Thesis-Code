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
    %e=y_goal(1:6)-y(1:6);
    [u,e]=lqr_control_tra(y(1:6),y_goal(1:6));
end

function [u,e]=attitude_control(y,y_goal)
    %u=[0;0;0];
    e=[0;0;0;0;0;0;0];
    u=quat_err_rate(y,y_goal);

end

%% Translation Helping Functions
function [u,e]=lqr_control_tra(y,y_goal)
    e=y_goal(1:6)-y(1:6);
    u=get_k_tra(y,[0;0;0])*(e);
    
end

function K=get_k_tra(y,u)
    %[A,B]=get_AB_tra(y,u);
    [A,B]=sat_jacobian(y,u);
    K=simple_lqr(A,B,get_Q_tra(),get_R_tra());
    %pause
    % K=[...
    % 1.85358394972458e-06     -2.06588662965874e-08      5.18850602740542e-21     0.000268762966446855      0.000771640310255927      2.03407214393561e-17
    % 7.98243596633682e-06     -4.42416220807265e-08     -1.14789168443559e-20      7.71640310255628e-05       0.00334912246918991      6.70750654550444e-20
    % 1.57635526707932e-19     -5.87692317116124e-22      8.68049091965597e-10      3.29266399252586e-16      7.33165003003798e-17      0.000263522147387399];
end

function [A,B]=get_AB_tra(y,u)
    %Init Vars using     %F_LV_LH
    p=Sat_params();
    n=sqrt(p.mu/p.rt^3);
    total_mass=10+p.mass;

    A = [0, 0, 0, 1, 0, 0 %
        0, 0, 0, 0, 1, 0 %
        0, 0, 0, 0, 0, 1 %
        3 * n ^ 2, 0, 0, 0, 2 * n, 0 %
        0, 0, 0, -2 * n, 0, 0 %
        0, 0, -n ^ 2, 0, 0, 0]; %

    B = [0, 0, 0
        0, 0, 0
        0, 0, 0
        1 / total_mass, 0, 0
        0, 1 / total_mass, 0
        0, 0, 1 / total_mass];

    A = A(1:6, 1:6);
    B = B(1:6, :);

end



function Q=get_Q_tra()
    Q=diag([3e1,2e1,1e1,0.1,0.1,0.1]*1e-2);
end
function R=get_R_tra()
    R=diag([1e1,1e2,1])*1e12;
end
function Return_lqr_values()
    [A,B]=get_AB_tra([0;0;0;0;0;0;10],[0;0;0]);
    K=lqr(A,B,get_Q_tra(),get_R_tra());
    %string([K,[";...";";...";";];"]])
    NoInput="No Input has been provided, the K lqr values are returned ";
    disp(NoInput)
    format longg
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





