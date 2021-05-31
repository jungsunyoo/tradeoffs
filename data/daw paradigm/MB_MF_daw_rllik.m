function LL = MB_MF_daw_rllik(x,subdata,opts)

% Likelihood function for Daw two-step paradigm in Kool, Cushman, &
% Gershman (2016).
%
% Depending on opts, the function reflects either a full hybrid model (opts.model = 1),
% a fully model-based model (opts.model = 2), or a model-free model
% (opts.model = 3). The fields opts.st and opts.respst determine the
% inclusion of stickiness parameters.
%
% Wouter Kool, Aug 2016 (based on code by Samuel J. Gershman)

y = zeros(1,10);
y(opts.ix==1) = x;

% switch opts.model
%     case 2   
%         y(4) = 1;
%     case 3
%         y(4) = 0;
% end
if ~opts.st
    y(5) = 0;
end
if ~opts.respst
    y(6) = 0;
end
if opts.polynomial<1
    y(7) = 0;
end
if opts.polynomial<2
    y(8) = 0;
end
if opts.beta==1
    y(9)=0;
end
if opts.alpha==1
    y(10)=0;
end

% parameters
b = y(1);           % softmax inverse temperature
lr = y(2);          % learning rate
lambda = y(3);      % eligibility trace decay
w0 = y(4);           % mixing weight
st = y(5);          % stickiness
respst = y(6);      % stickiness
w1 = y(7);
w2 = y(8);
b2 = y(9);
lr2 = y(10);

% initialization
Qd = zeros(3,2);            % Q(s,a): state-action value function for Q-learning
Tm = [.5 .5; .5 .5];        % transition matrix
M = [0; 0];                 % last choice structure
R = [0; 0];                 % last response structure

counts = zeros(2,2);        % counting transitions

N = size(subdata.choice1,1); %length(subdata.choice1);

LL = 0;

% loop through trials
for t = 1:N

    % Break if trial was missed
    if (subdata.choice1(t) == -1 || subdata.choice2(t) == -1)
        continue
    end
    
    state2 = subdata.state2(t)+1;
    
    if subdata.stim_1_left(t) == 2
        R = flipud(R);                                                          % arrange R to reflect stimulus mapping
    end
    
    maxQ = max(Qd(2:3,:),[],2);                                                 % optimal reward at second step
    Qm = Tm'*maxQ;                                                              % compute model-based value function
  
    
    if opts.model == 1
        if opts.polynomial > 0 %(opts.model ~= 3) && 
            w_ = w0 + ((t-N/2)*w1)/100 + (((t-N/2)/100)^2)*w2;
            w = 1/(1+exp(-w_));
        else
            w = w0;
        end      
    else
        if opts.polynomial == 2
            if t<round(N/3)
                w = w0;
            elseif (round(N/3) <=t) && (t<round(N/3)*2)
                w=w1;
            else
                w=w2;
            end
        elseif opts.polynomial == 1
            if t<round(N/2)
                w=w0;
            else
                w=w1;
            end
        else
            w=w0;
        end
    end
    
    
    Q = w*Qm + (1-w)*Qd(1,:)' + st.*M + respst.*R;                              % mix TD and model-based values
        
    LL = LL + b*Q(subdata.choice1(t))-logsumexp(b*Q);                           % update likelihoods
%     

    if opts.beta==1
        b_ = b;
    else
        b_ = b2;
    end
    
    LL = LL + b_*Qd(state2,subdata.choice2(t)) - logsumexp(b_*Qd(state2,:));

    M = [0; 0];
    M(subdata.choice1(t)) = 1;                                                  % make the last choice sticky
    
    R = zeros(2,1);
    if subdata.choice1(t) == subdata.stim_1_left(t)
        R(1) = 1;                                                               % make the last response sticky
    else
        R(2) = 1;
    end
    
    dtQ(1) = Qd(state2,subdata.choice2(t)) - Qd(1,subdata.choice1(t));          % backup with actual choice (i.e., sarsa)
    Qd(1,subdata.choice1(t)) = Qd(1,subdata.choice1(t)) + lr*dtQ(1);            % update TD value function
     
    dtQ(2) = subdata.win(t) - Qd(state2,subdata.choice2(t));                    % prediction error (2nd choice)

    
    if opts.alpha == 1
        lr_ = lr;
    else
        lr_  =lr2;
    end
    
    
    
    Qd(state2,subdata.choice2(t)) = Qd(state2,subdata.choice2(t)) + lr_*dtQ(2);  % update TD value function
    Qd(1,subdata.choice1(t)) = Qd(1,subdata.choice1(t)) + lambda*lr*dtQ(2);     % eligibility trace
    
    % pick the most likely transition matrix
    counts(subdata.state2(t),subdata.choice1(t)) = counts(subdata.state2(t),subdata.choice1(t))+1;
    
    if sum(diag(counts))>sum(diag(rot90(counts)))
        Tm = [.7 .3; .3 .7];        % transition matrix
    end
    if sum(diag(counts))<sum(diag(rot90(counts)))
        Tm = [.3 .7; .7 .3];        % transition matrix
    end
    if sum(diag(counts))==sum(diag(rot90(counts)))
        Tm = [.5 .5; .5 .5];        % transition matrix
    end
end

end
