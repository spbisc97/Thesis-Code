function [dy,u] = Sat_Attitude_Dyn(~,y,u) %#codegen
    %SAT_ATTITUDE_DYN attitude dynamic of the satellite
    %   The output of this function will be 7 elements
    %   quaternion+velocities

    parameters=Sat_params();
    
    g=9.81;

    q=y(1:4);
    w=y(5:7);

    %mass=y(8);
    u=min(parameters.Tmax,max(-parameters.Tmax,u(:)))*min(1,max(0,y(8)*1e2));
    torques=u;

    
    % assume a uniform density and a 10cm cube shape
    %calculated a priori in future implementations
    %if ~isnumeric(y(1))
    % syms Ix Iy Iz
    % I = diag([Ix, Iy, Iz]);
    %end

    %The gain K drives the norm of the quaternion state vector to 1.0 should εbecome nonzero.
    ep=1-sum(q.*q);
    K=0.1;%*norm(q);
    q_dot = 0.5 * omega(w)*q + K*ep*q;

    %Euler's equation of motion
    %(-cross(y(1:3), I * y(1:3))  from transport theorem
    %cross(w,ang_momentum), always zero if inertia is diagonal


    w_dot=[parameters.invI * (-cross(w(1:3), parameters.I * w(1:3)) + torques)]; %#ok

    m_dot=-sum(abs(u(:)))/(g*parameters.Isp);
    % this will be changed to a battery + rotation wheels


    dy=[q_dot;w_dot;m_dot];
end

% function mat=crossmat(a)
%     mat=[0,-a(3),a(2);a(3),0,-a(1);-a(2),a(1),0];
% end

function mat=omega(w)
    %mat=[0,-p,-q,-r;p,0,r,-q;q,-r,0,p;r,q,-p,0];
    mat=[0,-w(1),-w(2),-w(3);w(1),0,w(3),-w(2);w(2),-w(3),0,w(1);w(3),w(2),-w(1),0];
end

