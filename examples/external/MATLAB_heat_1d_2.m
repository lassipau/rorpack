function [Pval,CKRK,RLBL] = MATLAB_heat_1d_2(mode,cfunstr,freq,spgrid,K21funstr,L1funstr)
% Transfer function computations with Chebfun for the RORPack example case
% heat_1d_2
% The model is 
% w_t = (c(x)w_x)_x
% boundary conditions are 
% -w'(0,t)=u(t)+w_d(t) -- boundary input and input disturbance
% y(0)=w(0,t) -- collocated boundary observation
% w(1,t)=0 -- Dirichlet at x=1
% The system is exponentially stable and passive 
% 
% Parameters: 
% 'mode' - indicate whether to compute:
%   (1) 'Pvals' = transfer function value P(i*w_k)
%   (2) 'PKvals' = Stabilized transfer function value P(i*w_k) and CKRKvals
%   (3) 'PLvals' = Stabilized transfer function value P(i*w_k) and RLBLvals
% 'cfunstr' = the thermal diffusivity function given as a string with a
%   single variable 'x'.
% 'freq' = the frequency i*w_k where the values are computed
% 'spgrid' = the spatial grid where the CKRK and RLBL values are computed
%   (only required for modes (2) and (3))
% 'K21funstr' = function describing the feedback operator K21. Given as a 
%   string with a single variable 'x'.
% 'L1funstr' = function describing the output injection operator L1. Given as a 
%   string with a single variable 'x'.
%
% Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.


if isequal(mode,'Pvals')
        % Compute the transfer function P(s)=CR(s,A)B (now D=0)
        s = freq;

        cfun = str2func(['@(x)', cfunstr]);

        % Compute the transfer function P(s)
        A = chebop(0,1);
        A.op = @(x,w) s*w-diff(cfun(x)*diff(w));
        A.lbc = @(w) diff(w)+1;
        A.rbc = @(w) w;
        w = A\0;

        Pval = w(0);

        RLBL = [];
        CKRK = [];
        
elseif isequal(mode,'PKvals')
        % Compute P_K(s)=CR(s,A+B*K21)B and CR(s,A+B*K21) (now D=0)
        % Computation is done through the adjoint R(s,A)'*C'. This is
        % necessary to deal with the the point evaluation operator Cw=w(0).
        
        s = freq;

        cfun = str2func(['@(x)', cfunstr]);
        K21fun = chebfun(K21funstr,[0,1]);
        
        % Solve RC = R(s,A)'*C' and RK = R(s,A)'*K21'
        % First, R(s,A)'*C', where C' acts through the boundary condition
        A = chebop(0,1);
        A.op = @(x,w) conj(s)*w-diff(cfun(x)*diff(w));
        A.lbc = @(w) diff(w)+1;
        A.rbc = @(w) w;        
        RC = A\0;
        
        % Then compute R(s,A)'*K21', where K21' acts through the RHS
        A = chebop(0,1);
        A.op = @(x,w) conj(s)*w-diff(cfun(x)*diff(w));
        A.lbc = @(w) diff(w);
        A.rbc = @(w) w;
        RK = A\K21fun;
        
        % Now compute P_K(s) using the identity
       	% (P_K(s))' = (1-B'R(s,A)'*K21')^{-1}*B'R(s,A)'*C'
        Pval = conj(1/(1-RK(0))*RC(0));
        
        % Compute the adjoint of CR(s,A+B*K21) using the identity
        % (CR(s,A+B*K21))' = R(s,A)'*C' +
        % R(s,A)'*K21*(1-B'R(s,A)'K21')^{-1}*B'R(s,A)'*C'
        CKRKadjfun = RC + RK*(1-RK(0))^(-1)*RC(0);
        
        N = length(spgrid);
        CKRK = 1/(N-1)*CKRKadjfun(spgrid)';
        CKRK(1) = CKRK(1)/2;
        CKRK(end) = CKRK(end)/2;
        RLBL = [];

elseif isequal(mode,'PLvals')
        % Compute P_L(s)=CR(s,A+L1C)B and R(s,A+L1*C)B (now D=0)
        
        s = freq;

        cfun = str2func(['@(x)', cfunstr]);
        L1fun = chebfun(L1funstr,[0,1]);
        

        % Solve RB = R(s,A)B and RL = R(s,A)L1
        % First, R(s,A)B, where B acts through the boundary condition
        A = chebop(0,1);
        A.op = @(x,w) s*w-diff(cfun(x)*diff(w));
        A.lbc = @(w) diff(w)+1;
        A.rbc = @(w) w;        
        RB = A\0;
        
        % Then compute R(s,A)L1, where L1 acts through the RHS
        A = chebop(0,1);
        A.op = @(x,w) s*w-diff(cfun(x)*diff(w));
        A.lbc = @(w) diff(w);
        A.rbc = @(w) w;
        RL = A\L1fun;
        
        % Now compute P_L(s) using the identity
        % P_L(s)=(1-CR(s,A)L1)^{-1}CR(s,A)B
        Pval = 1/(1-RL(0))*RB(0);
        
        % Compute R(s,A+L1*C)B using the identity
        % R(s,A+L1*C)B=R(s,A)B+R(s,A)L1*(1-CR(s,A)L1)^{-1}*CR(s,A)B
        RLBLfun = RB + RL*(1-RL(0))^(-1)*RB(0);
        RLBL = RLBLfun(spgrid);
        CKRK = [];

end
