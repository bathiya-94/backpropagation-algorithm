clc;
clear;
load('Trained_Network_XOR.mat');

input = [0 0 ;
         1 1;
         1 0;
         0 1;];
     
 Z1 = W1 * input';
 A1 = logsig(Z1);
 Z2 = W2 *A1;        
 Y = logsig(Z2);
 disp(Y);
 