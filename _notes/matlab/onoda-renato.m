function Z_init = Onoda_12(Data,K)

[pcs] = pca(Data);

pcs = pcs(:,1:K);
pcs_norm = sum(pcs.^2).^0.5;
Data_norm = sum(Data.^2,2).^0.5;
scores=zeros(size(Data,1),K);

for i = 1 : size(Data,1)
    for k = 1 : K
       scores(i,k) = dot(Data(i,:), pcs(:,k)) / (pcs_norm(1,k) * Data_norm(i,1));
    end
end

[~, idx_min] = min(scores);

Z_init = Data(idx_min,:);

