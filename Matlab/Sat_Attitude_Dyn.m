function dy = Sat_Attitude_Dyn(~,y,u)
    %SAT_ATTITUDE_DYN attitude dynamic of the satellite
    %   The output of this function will be 7 elements
    %   velocities+ quaternion
    q=y(1:4);
    w=y(5:7);
    I = diag([0.0023, 0.0023, 0.0023]); % assume a uniform density and a 10cm cube shape
    invI=inv(I); %calculated a priori in future implementations

    %The gain K drives the norm of the quaternion state vector to 1.0 should εbecome nonzero.
    eps=1-sum(q.*q);
    K=10;%*norm(q);
    q_dot = 0.5 * [q(1)*eye(3) + crossmat(q(2:4)); -q(2:4)'] * w +K*eps*q;
    w_dot=[invI * (-cross(y(1:3), I * y(1:3)) + u)];

    dy=[q_dot;w_dot];
end

function mat=crossmat(a)
    mat=[0,-a(3),a(2);a(3),0,-a(1);-a(2),a(1),0];
end


