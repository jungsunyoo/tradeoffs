function results = mfit_wrapper(modelnumber)

% Function that fits behavioral data to reinforcement learning models
% for the novel two-step paradigm in Kool, Cushman, & Gershman (2016).
% 
% USAGE: results = mfit_wrapper
%
% NOTES:
%   This function requires the mfit model-fitting package: https://github.com/sjgershm/mfit
%
% Wouter Kool, Aug 2016

load groupdata

% addpath('/Users/yoojungsun0/Desktop/Repositories/mfit')
addpath('mfit_function')
opts.model = [1 2 3]; % 1 = hybrid model, 2 = model-based 3 = model-free
opts.st = [0 1]; % indexes presence of stimulus stickiness
opts.respst = [0 1]; % indexes presence of response stickiness
opts.polynomial = [0 1 2]; % jungsun added: polynomial function for w 

opts = factorial_models(opts);

nrstarts = 25;
% nrstarts=4;
nrmodels = length(opts);

data = groupdata.subdata(groupdata.i);

results = struct;

% run optimization
for m = modelnumber  %1:nrmodels
    
    disp(['Fitting model ',num2str(m)])
    [options, params] = set_opts(opts(m));
    f = @(x,data) MB_MF_novel_rllik(x,data,options);
%     results(m) = mfit_optimize(f,params,data,nstarts);
    m_ = mfit_optimize(f,params,data,nrstarts);
    results(m).nest = m_;
    results(m).opts = opts(m);
    savename = ['novel_model_', num2str(m)];
    save(savename, 'results');
    
%     results(m).opts = opts(model);
    
end
%save('novel_results_sum', 'results');
end
