clear;clc;
load groupdata
addpath('mfit_function')

data = groupdata.subdata(groupdata.i);

for isub = 1:length(data)
%     currRT1 = data(isub).rt1;
    currRT2 = data(isub).rt2;
%     half = round(length(currRT1)/2);
%     firstHalf = mean(currRT1(1:half));
%     secondHalf = mean(currRT1(half+1:length(currRT1)));
%     
%     RT1(isub,1) = firstHalf;
%     RT1(isub,2) = secondHalf;
    
%     RT1(isub,1) = mean(currRT1);



    half = round(length(currRT2)/2);
    firstHalf = mean(currRT2(1:half));
    secondHalf = mean(currRT2(half+1:length(currRT2)));
    
    RT2(isub,1) = firstHalf;
    RT2(isub,2) = secondHalf;
    
%     RT1(isub,1) = mean(currRT1);



%     RT2(isub,1) = mean(currRT2);
end