clc;
clear;
epochs =100;
input = [0 0;
         0 1;
         1 0;
         1 1;];
 
target = [ 0 1 1 1];
alpha = 0.1;

W1 = [0.1 -0.2;0.2 0.1;0.1 0.2];
W2 = [0.1 -0.1 0.2];

for i =1 : epochs
    for j =1 :length(input)
        Z1 = W1 * input(j,:)';
        A1 = logsig(Z1);
        Z2 = W2 *A1;        
        Y = logsig(Z2); 
        
        error = (target(j)-Y);
    
        %loss = error*error;

        dLoss_Y = -2*error;
        dLoss_Z2 = dLoss_Y * Y*(1-Y);
        dLoss_A1 = W2 * dLoss_Z2;
        dLoss_Z1 = dLoss_A1 * A1 *(1-A1);
        dLoss_W1 = dLoss_Z1 * input(i,:);
        dLoss_W2 = dLoss_Z2 * A1;

        W1 = W1 + alpha *dLoss_W1;
        W2 = W2 + alpha *dLoss_W2;
    end
end
