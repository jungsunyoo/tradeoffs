
clear;clc;
load('trial_results');
coef_x1 = [];
R2 = [];
for i = 1:197
    
   x1 = trial_results(i).Qvals_stage1(:,1) - trial_results(i).Qvals_stage1(:,2); % regressor of interest: absolute subjective value-difference
   x2 = trial_results(i).trialnum;
   x3 = trial_results(i).prevWin;
   x4 = trial_results(i).choice1 == trial_results(i).stim_s1_left;
   x5 = ones(length(x1),1);
   X = [x1, x2, x3, x4];
   y = log(trial_results(i).RT1);
   
   [b, bint, r, rint] = regress(y,X);
   coef_x1(i) = b(1);
%    R2(i) = stats(1);
    
end
mean(coef_x1)
% mean(R2)

%         x1 = np.array(val_diff) # 
%         x2 = np.array(np.arange(1,len(x1)+1)) # regressor 2: trial number 
%         x3_= np.array(r[rt2!=-1])
%         x3 = np.insert(x3_, 0, 0)[:-1] # regressor 3: if previous outcome was rewarded or no-reward
%         x4 = np.array(isLeft[rt2!=-1].astype(int)) # regressor 4: which button used (just Left)
%         x5 = np.zeros(len(val_diff)) # regressor 5: experience of each rocket pair
%         x6 = np.zeros(len(val_diff)) # regressor 6: experience of each chosen rocket
