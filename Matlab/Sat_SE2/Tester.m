clear all
close all
addpath("Auxiliary/")
addpath("Simulink/")
if ~exist("test","var")
    test=0;
    test_desc="" + ...
        "1) Linear and Nonlinear difference -" + ...
        "2) Difference btw integrators "
    prompt="Enter the test number: ";
    userinput=input(prompt);
    if ~isnumeric(userinput)
        test=1;
    else
        test=userinput;
    end
end
if test==1
    p=Sat_params();

    y0=1*1e3;

    initialconditions=[0 y0 0 y0/2000 0 0]';
    days=1;
    stepsize=0.05;
    timespan=1:3600*24*days;
    opts = odeset("MaxStep",stepsize);
    f=figure(1);
    hold off
    [t,w1]=ode45(@(t,w) Sat_dyn_Lin(t,w,[0,0],p),timespan,initialconditions,opts);
    plot3(w1(:,1),w1(:,2),t)
    hold on
    [t,w2]=ode45(@(t,w) Sat_dyn_nonLin(t,w,[0,0],p),timespan,initialconditions,opts);
    plot3(w2(:,1),w2(:,2),t)
    % nonLinear=1
    % [t,w2]=ode45(@(t,w) Sat_dyn(t,w,[0,0],p,nonLinear),timespan,initialconditions);


    plot3(0*t,0*t,t)

    legend
    grid on
    plot_settings(f)

    f=figure(2);
    plot(t,w1(:,1:2)-w2(:,1:2))
    grid on
    legend
    plot_settings(f)


    f=figure(3);
    plot(w2(:,1),w2(:,2),'DisplayName','w1')
    hold on
    plot(w1(:,1),w1(:,2),'DisplayName','w2')
    legend
    grid on
    xlabel("x")
    ylabel("y")
    plot_settings(f,"manual")


    f=figure(4);
    plot(t,[w2(:,3),w2(:,6)],"DisplayName","w2")
    hold on
    plot(t,sin(t/1000),"DisplayName","Torque")
    legend
    grid on
    plot_settings(f)

end
if test==2


    %Maximum torque on propulsor ~ 1e-3 N
    %Maximum momentum on rection wheel ~ 1e-2 Nm
    p=Sat_params();

    y0=5*1e3;

    initialconditions=[0 y0 pi/2 y0/2000 0 0]';
    days=1;
    stepsize=0.05;
    timespan05=1:stepsize:3600*24*days;
    timespan1=1:(stepsize*2):3600*24*days;
    %opts = odeset('MaxStep',stepsize*2); %avoid if possible,but tolerance is too low
    opts = odeset("RelTol",1e-6); %1e-8 seems to be enought, means ~ 1cm accuracy,
    %adding this to ode makes it much accurate but a little slower than rk4

    %I could try adding absolute tol
    timespan=timespan05;
    control=@(t) [1e-2;1e-2*sin(t/10000+pi/2)]' ;
    fh=@(t,w) Sat_dyn_Lin(t,w,control(t),p);
    tic
    [t1,w1]=ode45(fh,timespan,initialconditions,odeset("RelTol",1e-6)); %Dormand–Prince
    toc
    tic
    [t1,w5]=ode45(fh,timespan,initialconditions); %Dormand–Prince
    toc
    tic
    [t1,w4]=ode45(fh,timespan,initialconditions,odeset('MaxStep',stepsize*2)); %Dormand–Prince
    toc
    tic
    [t2,w2]=RK4(fh,timespan,initialconditions); 
    toc
    tic
    [t3,w3]=Euler(fh,timespan,initialconditions);
    toc

    % nonLinear=1
    % [t,w2]=ode45(@(t,w) Sat_dyn(t,w,[0,0],p,nonLinear),timespan,initialconditions);
    f=figure(2);
    plot(t1,w1(:,1),'--','DisplayName','odeRELTOL',LineWidth=2)
    hold on
    plot(t2,w2(:,1),'--','DisplayName','RK4')
    hold on
    plot(t3,w3(:,1),'--','DisplayName','EULER')
    hold on
    plot(t1,w4(:,1),'--','DisplayName','odeMaxStep')
    hold on
    plot(t1,w5(:,1),'--','DisplayName','ode',LineWidth=2)
    hold on
    
    grid on
    legend
    plot_settings(f)

    f=figure(5);
    plot(t1,w1(:,2),'--','DisplayName','odeRELTOL',LineWidth=2)
    hold on
    plot(t2,w2(:,2),'--','DisplayName','RK4')
    hold on
    plot(t3,w3(:,2),'--','DisplayName','EULER')
    hold on
    plot(t1,w4(:,2),'--','DisplayName','odeMaxStep')
    hold on
    plot(t1,w5(:,2),'--','DisplayName','ode',LineWidth=2)
    hold on



    grid on
    legend
    plot_settings(f)
    [M,I]=max(abs(w1(:,1:2)-w2(:,1:2)))
    [M,I]=max(abs(w1(:,1:2)-w3(:,1:2)))
    [M,I]=max(abs(w2(:,1:2)-w3(:,1:2)))

    f=figure(3);
    tiledlayout("flow")
    nexttile
    plot(t1,abs(w1(:,1:2)-w2(:,1:2)))
    nexttile
    plot(t1,abs(w1(:,1:2)-w3(:,1:2)))


    grid on
    legend
    plot_settings(f)


    f=figure(4);
    tiledlayout("flow")
    nexttile
    plot(t1,w1(:,3))
    nexttile
    plot(t2,w2(:,3))
    nexttile
    plot(t3,w3(:,3))


    grid on
    legend
    plot_settings(f)




end
if test==3
    %%test the docoupled system with LQR
    syms w [6,1]
    syms u [3,1]
    p=Sat_params();
    dw=Sat_dyn_Lin_Decoupled(0,w,u,p);
    A=eval(jacobian(dw,w));
    B=eval(jacobian(dw,u));

    k=lqr(A,B,eye(6)*1e-5,eye(1)*1e7);

    y0=5*1e3;

    initialconditions=[0 y0 pi y0/2000 0 0]';
    days=1;
    stepsize=0.05;
    timespan=1:stepsize:3600*24*days;

    %I could try adding absolute tol

    control=@(t,w) -k*w ;
    fh=@(t,w) Sat_dyn_Lin_Decoupled(t,w,control(t,w),p);

    tic
    [t1,w1]=Euler(fh,timespan,initialconditions);
    toc

    f=figure(2);
    plot(w1(:,1),w1(:,2),'--','DisplayName','w1')

    xlabel("x")
    ylabel("y")

    grid on
    legend
    plot_settings(f,"manual")



end
if test==4
    %simulink init
    global A B C D Q R N
    y0=5*1e3;
    syms w [6,1]
    syms u [3,1]
    p=Sat_params();
    initialconditions=[0 y0 pi y0/2000 0 0]';

    dw=Sat_dyn_Lin_Decoupled(0,w,u,p);
    A=eval(jacobian(dw,w));
    B=eval(jacobian(dw,u));
    
    clear w
    clear u
    
    [n,m]=size(B);
    C=eye(n);
    D=zeros(n,m);
    Q=eye(6);
    R=eye(3)*1e8;
    N=100;

    k=lqr(A,B,Q,R);
    


end
if test==5
        global A B C D Q R N

    y0=5*1e3;
    syms w [6,1]
    syms u [2,1]
    p=Sat_params();
    initialconditions=[0 y0 pi y0/2000 0 0]';

    dw=Sat_dyn_Lin(0,w,u,p);
    A=eval(jacobian(dw,w));
    A=subs(A,[u1,u2],[0,0]);
    B=eval(jacobian(dw,u));
    B=matlabFunction(B,'Vars',w3);
    
    [n,m]=size(B);
    C=eye(n);
    D=zeros(n,m);
    Q=eye(n);
    R=eye(m);
    N=100;

        
end
if test==6

    y0=5*1e3;
    syms w [6,1]
    syms u [2,1]
    p=Sat_params();
    initialconditions=[0 y0 pi y0/2000 0 0]';

    dw=Sat_dyn_Lin(0,w,u,p);

    V=1/2*((w1)^2 + (w2)^2 + (w3)^2 + (w4)^2 + (w5)^2 + (w6)^2);
    dV=w1*dw(1)+w2*dw(2)+w3*dw(3)+w4*dw(4)+w5*dw(5)+w6*dw(6);
end





function plot_settings(figure,aspect)
allAxesInFigure = findall(figure,'type','axes');
for i=1:length(allAxesInFigure)
    ax=allAxesInFigure(i);
    if exist("aspect","var")
        % Set the remaining axes properties
        set(ax,'DataAspectRatio',[1,1,1],'XLimitMethod','padded','YLimitMethod',...
            'padded','ZLimitMethod','padded');
    end
    box(ax,'on');
    grid(ax,'on');
    hold(ax,'off');

    % Create legend
    legend1 = legend(ax,'show');
end

end