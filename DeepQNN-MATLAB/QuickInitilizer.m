
function U = QuickInitilizer(M)

%This functions quickly initilizes a set of random unitaries for the
%Network with configuration M. 
%The unitaries for each layer are stored in the cell array U.
%U{k}(:,:,j) is the unitary acting on the jth neuron in the kth layer.

N_NumLay = size(M,2);



for k = 2:N_NumLay
    for j = 1:M(k)
       U{k}(:,:,j) = Randomunitary(2^(M(k-1)+1));
            
    end
end
