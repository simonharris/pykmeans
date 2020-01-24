function Z_init = Steinley_03(Data,K)

N = size(Data,1);

New_Z_init = zeros(K, size(Data,2));

SSE = inf;

for i = 1 : 5000

    U = randi(K,N,1);

    EmptyCluster = false;

    NewSSE = 0;

    for k = 1 : K

        if sum(U==k)==0, EmptyCluster=true;end

        New_Z_init(k,:) = mean(Data(U==k,:),1);

        NewSSE = NewSSE + sum(sum((Data(U==k,:) - New_Z_init(k,:)).^2,2));

    end

    if EmptyCluster, continue;end

    if NewSSE < SSE

        Z_init = New_Z_init;

        SSE = NewSSE;

        Z_init = New_Z_init;

    end

end
