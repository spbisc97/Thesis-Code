function K = simple_lqr(A,B,Q,R)
%myFun - Description
%
% Syntax: k = simple_lqr(input)
%
% Long description
    S=[];   
    E=[];
    G=[];
    
    [~,K]=icare(A, B, Q, R, S, E, G);
    
    % n=6;Ts=1;
    % Ad = (eye(n) + A * Ts / 2.) * pinv(eye(n) - A * Ts / 2.); %tustin, bilinear trans
    % Bd = pinv(A) * (Ad - eye(n)) * B;
    % [~,K]=idare(Ad,Bd,Q,R,S,E);
    
end