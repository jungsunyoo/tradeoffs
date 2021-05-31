function results = mfit_wrapper_pure(modelnum)

% Function that fits behavioral data to reinforcement learning models
% for the Daw two-step paradigm in Kool, Cushman, & Gershman (2016).
% 
% USAGE: results = mfit_wrapper
%
% NOTES:
%   This function requires the mfit model-fitting package: https://github.com/sjgershm/mfit
%
% Wouter Kool, Aug 2016

load groupdata
addpath('mfit_function')
% addpath('/Users/yoojungsun0/Desktop/Repositories/mfit')

% opts.model = [1 2 3]; % 1 = hybrid model, 2 = model-based 3 = model-free
opts.st = [0 1]; % indexes presence of stimulus stickiness
opts.respst = [0 1]; % indexes presence of response stickiness
opts.alpha = [1 2]; % whether to use one or two lr
opts.beta = [1 2]; % whether to use one or two beta
opts.gamma = [0 1];
% opts.polynomial = [0 1 2]; % jungsun added: polynomial function for w 
opts.model = [1 2]; % 1 = fully MB, 2 = fully MF

opts = factorial_models(opts);

nrstarts = 25;
% nrstarts=4;
nrmodels = length(opts);

data = groupdata.subdata(groupdata.i);

results = struct;

% run optimization
for m = modelnum%1:nrmodels
    
    disp(['Fitting model ',num2str(m)])
    
%     if (opts(m).polynomial==0) && (opts(m).model==2)
%         % no need to run because overlap
%         disp('No need to run this model')
%         break
%     end    
%     
    [options, params] = set_opts_pure(opts(m));
    f = @(x,data) pure_MB_MF_daw_rllik(x,data,options);
    m_ = mfit_optimize(f,params,data,nrstarts);
    results(m).nest = m_;
    results(m).opts = opts(m);
    savename = ['daw_model_pure_', num2str(m)];
    save(savename, 'results');
    
end

% save('daw_results_sum', 'results');

end
