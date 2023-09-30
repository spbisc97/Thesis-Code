function dw=Sat_dyn_Lin(~,w,u,p)
%Satellite linear dynamics
arguments
    ~
    w (6,1)
    u (2,1)
    p 
end

%u is a 3x1 vector containing the following control inputs
%u(1)=F_LV_LH
%u(2)=F_Rotoation

%w is a 6x1 vector containing the following states
%w(1)=x
%w(2)=y
%w(3)=phi
%w(4)=x_dot
%w(5)=y_dot
%w(6)=phi_dot

%p is a struct containing the following parameters
%p.mu=Gravitational parameter
%p.rt=Radius of target
%p.mass=Mass of satellite
%p.Tmax=Maximum thrust
%p.Tmin=Minimum thrust



%check direction of arrays
w=w(:);
dw=w;%zeros(6,1);
u=u(:);

%Satellite dynamics
x=w(1);
y=w(2);
phi=w(3);
x_dot=w(4);
y_dot=w(5);
phi_dot=w(6);
mass=p.mass;

%z=0;


%Takes u directly as acc
%u=min(p.Tmax,max(-p.Tmax,u))*min(1,max(0,y(7)*1e2));
torques=[sin(phi); cos(phi);0]*u(1)+[0;0;1]*u(2);



%y 1to6 position and velocity
%y 7 is mass

%rc=((p.rt+x)^2+y^2+z^2)^(1/2);
n=sqrt(p.mu/p.rt^3);
%eta=p.mu/(rc^3);

%Init Vars using     %F_LV_LH
dw(1)=x_dot;
dw(2)=y_dot;
dw(3)=phi_dot;
dw(4)=(3*n^2*x)+(2*n*y_dot)+(torques(1)/mass);
dw(5)=(-2*n*x_dot)+(torques(2)/mass);
dw(6)=p.invI*torques(3)/mass;


end


