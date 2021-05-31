
clear;clc;
load('results.mat');
y = results(1).x;

load groupdata
data = groupdata.subdata(groupdata.i);

currdir = '/Users/yoojungsun0/Desktop/Repositories/tradeoffs/data/daw paradigm';
resultsdir = fullfile(currdir, filesep, 'trialdata');
if ~exist(resultsdir)
    mkdir(resultsdir)
end
% st=0;
subInd=[];
w = y(:,4);

for isub = 1:length(data)
    subdata = data(isub);
    N = size(subdata.choice1,1); %length(subdata.choice1);
    
    counts = zeros(2,2);        % counting transitions    
    
    % parameters
    b = y(isub,1);           % softmax inverse temperature
    lr = y(isub,2);          % learning rate
    lambda = y(isub,3);      % eligibility trace decay
    w0 = y(isub,4);           % mixing weight
    st = 0;
    respst = 0;   
    
    
%     if lr < 0.1
%         continue % if lr is too low, can't estimate trial-by-trial estimate
%     end
    
%     st=st+1;
    
    % initialization
    Qd = zeros(3,2);            % Q(s,a): state-action value function for Q-learning
    Tm = [.5 .5; .5 .5];        % transition matrix
    M = [0; 0];                 % last choice structure
    R = [0; 0];                 % last response structure
%     LL = 0;
    Qvals_stage1 = [];
    Qvals_stage2 = [];
    Qmb_stage1 = [];
    Qmf_stage1 = [];
    ValV = [];
    PV = [];
    A1 = [];
    A2 = [];
    probs_stage1 = [];
    probs_stage2 = [];
    RT1=[];
    RT2=[];
    
    prevWin = [];
    prevChoice1 = [];
    
    choice1 = [];
    stim_s1_left = [];
    trialnum = [];
    
    % loop through trials
    ct = 0;
    for t = 1:N
        
        % Break if trial was missed
        if (subdata.choice1(t) == -1 || subdata.choice2(t) == -1)
            continue
        end
        ct = ct+1;
        A1(ct) = subdata.choice1(t);
        A2(ct) = subdata.choice2(t);
        state2 = subdata.state2(t)+1;
        RT1(ct,1) = subdata.rt1(t);
        RT2(ct,1) = subdata.rt2(t);
        
        prevWin(ct,1) = subdata.prevwin(t);
        prevChoice1(ct,1) = subdata.prevchoice1(t);
        choice1(ct,1) = subdata.choice1(t);
        stim_s1_left(ct,1) = subdata.stim_1_left(t);
        
        trialnum(ct,1) = ct;

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

        w=w0;


        Q = w*Qm + (1-w)*Qd(1,:)';                              % mix TD and model-based values
        Qvals_stage1(ct,:) = Q';
        Qmf_stage1(ct,:) = Qd(1,:);
        Qmb_stage1(ct,:) = Qm';
        
        Qvals_stage2(ct,:) = Qd(state2,:);
%         LL = LL + b*Q(subdata.choice1(t))-logsumexp(b*Q);                           % update likelihoods
%         LL = LL + b*Qd(state2,subdata.choice2(t)) - logsumexp(b*Qd(state2,:));


        dtQ(1) = Qd(state2,subdata.choice2(t)) - Qd(1,subdata.choice1(t));          % backup with actual choice (i.e., sarsa)
        Qd(1,subdata.choice1(t)) = Qd(1,subdata.choice1(t)) + lr*dtQ(1);            % update TD value function

        dtQ(2) = subdata.win(t) - Qd(state2,subdata.choice2(t));                    % prediction error (2nd choice)

        Qd(state2,subdata.choice2(t)) = Qd(state2,subdata.choice2(t)) + lr*dtQ(2);  % update TD value function
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


% 
%     trial_results(isub).Qvals_stage1 = Qvals_stage1;
%     trial_results(isub).Qvals_stage2 = Qvals_stage2;
%     
%     trial_results(isub).Qmb_stage1 = Qmb_stage1;
%     trial_results(isub).Qmf_stage1 = Qmf_stage1;
% 
% 
% %     trial_results(isub).PV = PV;
% %     trial_results(isub).probs_stage1 = probs_stage1;
% %     trial_results(isub).probs_stage2 = probs_stage2;
%     
%     trial_results(isub).RT1 = RT1;
%     trial_results(isub).RT2 = RT2;
%     
%     trial_results(isub).prevWin = prevWin;
%     trial_results(isub).prevChoice1 = prevChoice1;
%     trial_results(isub).choice1 = choice1;
%     trial_results(isub).stim_s1_left = stim_s1_left;
%     
%     trial_results(isub).trialnum = trialnum;
%     
    isLeft = stim_s1_left == choice1;
%         savename = ['model3_trialdata_', filenames{i}, '.txt'];
%         savedata = [trial_data.Qvals_stage1 trial_data.Qvals_stage2 trial_data.PV trial_data.Qmb_stage1 trial_data.Qmf_stage1];

    valuetable = [Qvals_stage1 Qvals_stage2 Qmb_stage1 Qmf_stage1];
    datatable = [trialnum prevWin isLeft RT1 RT2]; 
    savename_val = ['model1_trialdata_value_sub', num2str(isub), '.txt'];
    savename_data = ['model1_trialdata_data_sub', num2str(isub), '.txt'];
    if lr>0.1
        csvwrite(fullfile(resultsdir, filesep, savename_val), valuetable);
        csvwrite(fullfile(resultsdir, filesep, savename_data), datatable);
        subInd = [subInd; isub];
    end
    
end
csvwrite(fullfile(currdir, filesep, 'valid_subind.txt'), subInd)
csvwrite(fullfile(currdir,filesep,'w.txt'),w)
