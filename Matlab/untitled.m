% Define initial state
q0 = [1, 0, 0, 0]'; % initial quaternion
omega = [0.5, 0.2, -0.1]'; % initial angular velocity

% Define time span and time step
tspan = linspace(0, 10, 1000);
dt = tspan(2) - tspan(1);

% Integrate quaternion dynamics using ode45
[~, q] = ode45(@(t, q) attitude_dynamics(t, q, omega), tspan, q0);

% Define cube vertices in local coordinates
verts_local = [-1 -1 -1;
                1 -1 -1;
                1  1 -1;
               -1  1 -1;
               -1 -1  1;
                1 -1  1;
                1  1  1;
               -1  1  1];

% Define cube faces
faces = [1 2 3 4;
         2 6 7 3;
         6 5 8 7;
         5 1 4 8;
         1 2 6 5;
         4 3 7 8];

% Define rotation matrix to rotate from local to global coordinates
R = @(q) [q(1)^2 + q(2)^2 - q(3)^2 - q(4)^2, 2*(q(2)*q(3) - q(1)*q(4)),     2*(q(2)*q(4) + q(1)*q(3));
          2*(q(2)*q(3) + q(1)*q(4)),     q(1)^2 - q(2)^2 + q(3)^2 - q(4)^2, 2*(q(3)*q(4) - q(1)*q(2));
          2*(q(2)*q(4) - q(1)*q(3)),     2*(q(3)*q(4) + q(1)*q(2)),     q(1)^2 - q(2)^2 - q(3)^2 + q(4)^2];

% Initialize figure
fig = figure;
axis equal
axis([-2 2 -2 2 -2 2])
grid on
xlabel('X')
ylabel('Y')
zlabel('Z')

% Animate cube rotation
for i = 1:length(q)
    % Rotate cube vertices to global coordinates
    verts_global = (R(q(i,:))*verts_local')';
    
    % Update cube vertices in plot
    h.Vertices = verts_global;
    
    % Redraw plot
    drawnow
    
    % Pause for animation
    pause(dt)
end

% Define quaternion dynamics function
function q_dot = attitude_dynamics(t, q, w)
    q_dot = 0.5 * [q(1)*eye(3) + crossmat(q(2:4)); -q(2:4)'] * w;
end

% Define quaternion multiplication function
function q = qmult(q1, q2)
    q = [q1(1)*q2(1) - dot(q1(2:4), q2(2:4)); ...
         q1(1)*q2(2:4) + q2(1)*q1(2:4) + cross(q1(2:4), q2(2:4))];
end

function C = crossmat(v)
% Create a 3x3 cross product matrix from a 3-element vector
C = [    0, -v(3),  v(2);
      v(3),     0, -v(1);
     -v(2),  v(1),    0];
end

