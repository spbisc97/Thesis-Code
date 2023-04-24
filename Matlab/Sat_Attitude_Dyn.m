function dy = Sat_Attitude_Dyn(~,y,u) %#codegen
    %SAT_ATTITUDE_DYN attitude dynamic of the satellite
    %   The output of this function will be 7 elements
    %   quaternion+velocities


    parameters.Isp=2250;
    parameters.Tmax=1.05e-7;
    g=9.81;


    q=y(1:4);
    w=y(5:7);
    I = diag([0.0023, 0.0023, 0.0023]); % assume a uniform density and a 10cm cube shape
    invI=inv(I); %calculated a priori in future implementations
    %if ~isnumeric(y(1))
    % syms Ix Iy Iz
    % I = diag([Ix, Iy, Iz]);
    %end

    %The gain K drives the norm of the quaternion state vector to 1.0 should εbecome nonzero.
    eps=1-sum(q.*q);
    K=0.1;%*norm(q);
    q_dot = 0.5 * omega(w)*q + K*eps*q;

    %Euler's equation of motion
    %(-cross(y(1:3), I * y(1:3))  from transport theorem
    %cross(w,ang_momentum), always zero if inertia is diagonal


    w_dot=[invI * (-cross(w(1:3), I * w(1:3)) + u)];

    m_dot=-sum(abs(u(:)))/(g*parameters.Isp);


    dy=[q_dot;w_dot;m_dot];
end

function mat=crossmat(a)
    mat=[0,-a(3),a(2);a(3),0,-a(1);-a(2),a(1),0];
end

function mat=omega(w)
    %mat=[0,-p,-q,-r;p,0,r,-q;q,-r,0,p;r,q,-p,0];
    mat=[0,-w(1),-w(2),-w(3);w(1),0,w(3),-w(2);w(2),-w(3),0,w(1);w(3),w(2),-w(1),0];
end

