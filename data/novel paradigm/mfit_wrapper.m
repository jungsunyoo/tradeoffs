function results = mfit_wrapper(modelnum)

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
opts.st = [0 1]; % indexes presence of stimulus stickiness
opts.respst = [0 1]; % indexes presence of response stickiness
opts.polynomial = [0 1 2]; % jungsun added: polynomial function for w 

opts.model = [1 2]; % polynomial, window
opts = factorial_models(opts);

nrstarts = 25;
% nrstarts=4;
nrmodels = length(opts);

data = groupdata.subdata(groupdata.i);

results = struct;

% run optimization
for m = modelnum%1:nrmodels
    
    disp(['Fitting model ',num2str(m)])

    if (opts(m).polynomial==0) && (opts(m).model==2)
        % no need to run because overlap
        disp('No need to run this model')
        break
    end        
    
    [options, params] = set_opts(opts(m));
    f = @(x,data) MB_MF_novel_rllik(x,data,options);
%     results(m) = mfit_optimize(f,params,data,nstarts);
    m_ = mfit_optimize(f,params,data,nrstarts);
    results(m).nest = m_;
    results(m).opts = opts(m);
    savename = ['novel_model_hybrid_', num2str(m)];
    save(savename, 'results');
    
%     results(m).opts = opts(model);
    
end
end
