%Trains the Network and gives out the array with all trained unitaries in U
%and an array CList with all Cost functions while training
function [U,CList] = TrainNetwork(phi_in,phi_out,U,M,lambda,iter)

%iter: number of iterations the network trains (update all unitaries)
eps= 0.1;

N_NumTrain = size(phi_in,2);
N_NumLay = size(M,2);

CList = [CostNetwork(phi_in,phi_out,U,M)];





%Update the Unitaries iter times

for round=2:iter
    K ={};
    %Generating all K Update Matrices
    for x = 1:N_NumTrain
        
    for k = 2:N_NumLay
        %Initilize a state to calculate state of left side of the Commutator in the Update Matrix M
        %i.e. the one coming from the \mathcal{E} or "ApplyLayer" Channel.
        if x == 1
           K{k} = zeros(2^(M(k-1)+1),2^(M(k-1)+1),M(k));
        end
        if k == 2
           rho_left_prev = phi_in(:,x)*phi_in(:,x)';
        else
        rho_left_prev = ApplyLayer(rho_left_prev,U{k-1},M(k-1),M(k-2));
        end
        rho_left =  kron(rho_left_prev,[1;zeros(2^M(k)-1,1)]*[1;zeros(2^M(k)-1,1)]');
        
      for j = 1:M(k)
          %Initilize a state to calculate the state of the right hand side
          %of the Commutator in the Update Matrix M, i.e. the one coming
          %from the conjugate F Channel.
          if k==2 && j==1 
          for k_1 = 2:N_NumLay
            k_2 = N_NumLay -k_1 +2;
            if k_2 == N_NumLay
                rho_right_prev = phi_out(:,x)*phi_out(:,x)';
            else
               rho_right_prev = FChannel(rho_right_prev,U{k_2+1},M(k_2 +1),M(k_2));
            end
            rho_right{k_2} = kron(eye(2^M(k_2-1)),rho_right_prev);
            for j_1 = 1:M(k_2)
                j_2 = M(k_2) -j_1 +1;
                V = Swap(kron(U{k_2}(:,:,j_2),eye(2^(M(k_2)-1))),[M(k_2-1)+1,M(k_2-1)+j_2],2*ones(1,M(k_2-1)+M(k_2)));
               rho_right{k_2} = V'*rho_right{k_2}*V;
            end
          end
          end
               
      %Generating left hand side of commutator for M_j^k. Note that we can use application
      %of all unitaries before the _j^k Neuron
               V = Swap(kron(U{k}(:,:,j),eye(2^(M(k)-1))),[M(k-1)+1,M(k-1)+j],2*ones(1,M(k-1)+M(k)));
               rho_left = V*rho_left*V';
              
               rho_right{k} = V*rho_right{k}*V';
               
               M_Update = Comm(rho_left,rho_right{k});
                            
               K{k}(:,:,j) = K{k}(:,:,j) + PartialTrace(M_Update,[M(k-1)+1:M(k-1)+j-1,M(k-1)+j+1:M(k-1)+M(k)] , 2*ones(1,M(k-1)+M(k))); 

        end
          
           
    end  
      
    end
     
    

%Updating all Unitaries in the Network
for k = 2:N_NumLay
     for j = 1:M(k)
          U{k}(:,:,j) =expm((-eps*2^M(k-1)/(N_NumTrain*lambda))*K{k}(:,:,j))*U{k}(:,:,j);
      end
end

%Save the Costfunction of this round
CList(round) = CostNetwork(phi_in,phi_out,U,M);

end

end
