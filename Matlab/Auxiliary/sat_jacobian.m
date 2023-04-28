function [A, B] = sat_jacobian(y, u)
    % [A,B]=sat_jacobian(y,u);
    % Jacobian of satellite translational dynamics
    p = Sat_params();
    total_mass = p.mass + p.fuel_mass;

    sigma = 1 / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2);

    A = [0,                                                                0,                                                      0,                  1,                 0, 0
         0,                                                                0,                                                      0,                  0,                 1, 0
         0,                                                                0,                                                      0,                  0,                 0, 1
         p.mu / p.rt^3 - p.mu / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(3 / 2) + 3 * p.mu * sigma * (p.rt + y(1))^2,                                          3 * p.mu * sigma * y(2) * (p.rt + y(1)),                                3 * p.mu * sigma * y(3) * (p.rt + y(1)),                  0, 2 * (p.mu / p.rt^3)^(1 / 2), 0
         3 * p.mu * sigma * y(2) * (p.rt + y(1)), p.mu / p.rt^3 - p.mu / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(3 / 2) + 3 * p.mu * sigma * y(2)^2,                                       3 * p.mu * sigma * y(2) * y(3), -2 * (p.mu / p.rt^3)^(1 / 2),                 0, 0
         3 * p.mu * sigma * y(3) * (p.rt + y(1)),                                                 3 * p.mu * sigma * y(2) * y(3), 3 * p.mu * sigma * y(3)^2 - p.mu / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(3 / 2),                  0,                 0, 0];

    B = [0,            0,            0
         0,            0,            0
         0,            0,            0
         1 / total_mass,            0,            0
         0, 1 / total_mass,            0
         0,            0, 1 / total_mass];

end

% A = [0, 0, 0, 1, 0, 0
%          0, 0, 0, 0, 1, 0
%          0, 0, 0, 0, 0, 1
%          p.mu / p.rt^3 - p.mu / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(3 / 2) + (3 * p.mu * (2 * p.rt + 2 * y(1)) * (p.rt + y(1))) / (2 * ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2)), (3 * p.mu * y(2) * (p.rt + y(1))) / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2), (3 * p.mu * y(3) * (p.rt + y(1))) / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2), 0, 2 * (p.mu / p.rt^3)^(1 / 2), 0
%          (3 * p.mu * y(2) * (2 * p.rt + 2 * y(1))) / (2 * ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2)), p.mu / p.rt^3 - p.mu / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(3 / 2) + (3 * p.mu * y(2)^2) / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2), (3 * p.mu * y(2) * y(3)) / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2), -2 * (p.mu / p.rt^3)^(1 / 2), 0, 0
%          (3 * p.mu * y(3) * (2 * p.rt + 2 * y(1))) / (2 * ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2)), (3 * p.mu * y(2) * y(3)) / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2), (3 * p.mu * y(3)^2) / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(5 / 2) - p.mu / ((p.rt + y(1))^2 + y(2)^2 + y(3)^2)^(3 / 2), 0, 0, 0];

%     B = [0,            0,            0
%          0,            0,            0
%          0,            0,            0
%          1 / total_mass,            0,            0
%          0, 1 / total_mass,            0
%          0,            0, 1 / total_mass];