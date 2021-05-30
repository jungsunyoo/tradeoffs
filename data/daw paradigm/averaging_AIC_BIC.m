clear;clc;
nmodel = 24;
avg_aic=0;
avg_bic=0;
for m = 1:nmodel
    filename = ['daw_model_hybrid_', num2str(m), '.mat'];
    if exist(filename)
        load(filename);
        avg_aic(m) = mean(results(m).nest.aic);
        avg_bic(m) = mean(results(m).nest.bic);
    else
        avg_aic(m) =9999;
        avg_bic(m) =9999;
    end
end
avg_aic = avg_aic';
avg_bic = avg_bic';