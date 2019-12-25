clc;
clear;

epochs =100000;

input = [0 0;
         0 1;
         1 0;
         1 1;];
 
target = [ 0 1 1 1];

alpha = 0.1;
%Learning rate

W1 = 2*rand(3,2)-1;
% Randomly initialize  input layer weights

W2 =  2*rand(1,3)-1;
% Ramdonly  initialize output layer weights

loss = inf;
iter = 0;

while(loss >= 0.001 && iter < epochs)
%for i =1 : epochs
    %Forward propation
        Z1 = W1 * input'; 
        %Weighted sum of the hidden layer
        A1 = logsig(Z1);
        % Activating the neuron using sigmoid function
        Z2 = W2 *A1; 
        %Weighted sum of the output layer
        Y = logsig(Z2);
         % Activating the output using sigmoid function
        
        error = (target-Y);
        % Calculating the error function
        
        loss = sum(error.*error)/2;
    %End of forward propagation    
    
    %Back propagation      
        dLoss_Y = -error;
        dLoss_Z2 = dLoss_Y .* (Y.*(1-Y));

        dLoss_A1 = W2' * dLoss_Z2;
        dLoss_W2 = dLoss_Z2 * A1';

        dLoss_Z1 = dLoss_A1 .*( A1 .*(1-A1));
        dLoss_W1 = dLoss_Z1 * input;
    %End of  backpropagation    

    %Updating the weights
        W1 = W1 - alpha *dLoss_W1;
        W2 = W2 - alpha *dLoss_W2;
        
        iter = iter + 1;    
end
save('Trained_Network_OR.mat')