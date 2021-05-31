function [opts, param] = set_opts_pure(opts)

% Code that sets up different options, and empirical priors for model-fitting
% procedure for the Daw two-step paradigm in Kool, Cushman, & Gershman (2016).
% Parameters of the prior are chosen after Gershman (2016).
%
% Wouter Kool, Aug 2016

opts.ix = ones(1,8);

lb = 0;
ub = 1;

if ~opts.st, opts.ix(4) = 0; end
if ~opts.respst, opts.ix(5) = 0; end
if opts.gamma==0, opts.ix(6)=0;end
if opts.beta==1, opts.ix(7)=0;end
if opts.alpha==1, opts.ix(8)=0;end

% create parameter structure
g = [4.82 0.88];  % parameters of the gamma prior
param(1).name = 'inverse temperature';
param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(1).lb = 0;   % lower bound
param(1).ub = 20;  % upper bound

param(2).name = 'learning rate';
param(2).logpdf = @(x) 0;
param(2).lb = lb;
param(2).ub = ub;

param(3).name = 'eligibility trace decay';
param(3).logpdf = @(x) 0;
param(3).lb = lb;
param(3).ub = ub;

mu = 0.15; sd = 1.42;   % parameters of choice stickiness

param(4).name = 'choice stickiness';
param(4).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(4).lb = -20;
param(4).ub = 20;

mu = 0.15; sd = 1.42;    % parameters of response stickiness
param(5).name = 'response stickiness';
param(5).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(5).lb = -20;
param(5).ub = 20;

param(6).name = 'memory decay';
param(6).logpdf = @(x) 0;
param(6).lb = lb;
param(6).ub = ub;

g = [4.82 0.88];  % parameters of the gamma prior
param(7).name = 'inverse temperature 2';
param(7).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(7).lb = 0;   % lower bound
param(7).ub = 20;  % upper bound

param(8).name = 'learning rate 2';
param(8).logpdf = @(x) 0;
param(8).lb = lb;
param(8).ub = ub;

param = param(opts.ix==1);

end
