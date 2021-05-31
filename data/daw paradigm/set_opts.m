function [opts, param] = set_opts(opts)

% Code that sets up different options, and empirical priors for model-fitting
% procedure for the Daw two-step paradigm in Kool, Cushman, & Gershman (2016).
% Parameters of the prior are chosen after Gershman (2016).
%
% Wouter Kool, Aug 2016

opts.ix = ones(1,10);

lb = 0;
ub = 1;
w_lb = -10;
w_ub = 10;

if (opts.model==2) || (opts.polynomial==0) % if window model

    w_lb=lb;
    w_ub = ub;

end
% if opts.model==3, opts.ix(4) = 0; end
if ~opts.st, opts.ix(5) = 0; end
if ~opts.respst, opts.ix(6) = 0; end
if opts.polynomial<2, opts.ix(8) = 0; end
if opts.polynomial<1, opts.ix(7) = 0; end
if opts.beta==1, opts.ix(9)=0;end
if opts.alpha==1, opts.ix(10)=0;end


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
if (opts.model==2) || (opts.polynomial==0) % if window model or single w
    w_pdf =  @(x) 0;
else
    w_pdf =  @(x) sum(log(normpdf(x,mu,sd)));
end



param(4).name = 'w0';
param(4).logpdf = w_pdf;
param(4).lb = w_lb;
param(4).ub = w_ub;

% mu = 0.15; sd = 1.42;   % parameters of choice stickiness
param(5).name = 'choice stickiness';
param(5).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(5).lb = -20;
param(5).ub = 20;

mu = 0.15; sd = 1.42;    % parameters of response stickiness
param(6).name = 'response stickiness';
param(6).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(6).lb = -20;
param(6).ub = 20;

param(7).name = 'w1';
param(7).logpdf = w_pdf;
param(7).lb = w_lb;
param(7).ub = w_ub;

param(8).name = 'w2';
param(8).logpdf = w_pdf;
param(8).lb = w_lb;
param(8).ub = w_ub;



g = [4.82 0.88];  % parameters of the gamma prior
param(9).name = 'inverse temperature 2';
param(9).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(9).lb = 0;   % lower bound
param(9).ub = 20;  % upper bound

param(10).name = 'learning rate 2';
param(10).logpdf = @(x) 0;
param(10).lb = lb;
param(10).ub = ub;




param = param(opts.ix==1);

end
