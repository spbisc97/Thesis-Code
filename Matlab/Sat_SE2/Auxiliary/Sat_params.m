
function parameters=Sat_params() %maybe should be a class?
%% Sat Parameters

%%Satellite Motor parameters based on busek bit-1
%satellite Isp consumption
parameters.Isp=2250;
%max trust
%parameters.Tmax=1.05e-5;
paramentrs.Tmax=1.05e-3;

%inertia matrix, and its inverse
%Ix = 8.33e-2;
%Iy = 1.08e-1;
%Iz = 4.17e-2; %thats the only one i need on SE2
parameters.I=4.17e-2;
parameters.invI=23.9808;

parameters.mass=30;%kg
parameters.fuel_mass=10;%kg

parameters.mu=3.986004418*10^14;
parameters.rt=6.6*1e6;
%parameters.n= 1.2e-3;%=sqrt(mu/rt^3);

%% when I need symbolic
% syms Ix Iy Iz mass Isp Tmax mu rt
% parameters.I=diag([Ix,Iy,Iz]);
% parameters.invI=diag([1/Ix,1/Iy,1/Iz]);
% parameters.mass=mass;
% parameters.Isp=Isp;
% parameters.Tmax=Tmax;
% parameters.mu=mu;
% parameters.rt=rt;
% y=sym('y',[7 1]);u=sym('u',[3 1])


end