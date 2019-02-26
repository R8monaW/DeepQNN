%Calculates Costfunctions for Noisy data Plot. phi_in_start and
%phi_out_start should be RandomUnitaryTrainingData

function CList = Noisydata(phi_in_start,phi_out_start,U,M,s)
%s is the number of steps between each point

N_NumTrain = size(phi_in_start,2);

dim = 2^M(1);


N_iter = N_NumTrain/s;
% N_iter should be integer


CList = zeros(1,N_iter);
phi_in = phi_in_start;
phi_out = phi_out_start;
[phi_in_noisy,phi_out_noisy] = Randomtrainingdata(N_NumTrain,dim);

for j = 1:N_iter+1
    
    %Specify good iteration number and lambda
    x = TrainNetwork(phi_in,phi_out,U,M,1.5,400);
    CList(j) = CostNetwork(phi_in_start,phi_out_start,x,M);
    if j < N_iter+1
        phi_in = phi_in_start(:,randperm(N_NumTrain));
        phi_out = phi_out_start(:,randperm(N_NumTrain));
        phi_in_noisy = phi_in_noisy(:,randperm(N_NumTrain));
        phi_out_noisy = phi_out_noisy(:,randperm(N_NumTrain));
        phi_in(:,1:s*j) = phi_in_noisy(:,1:s*j);
        phi_out(:,1:s*j) = phi_out_noisy(:,1:s*j);
        phi_in = phi_in(:,randperm(N_NumTrain));
        phi_out = phi_out(:,randperm(N_NumTrain));
    end
    
end


 
end
