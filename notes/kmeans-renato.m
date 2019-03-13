function U = kmeans123(Data, K)

    [N,M] = size(Data);

    % Randomly initialise Z
    Z = Data(randsample(N,K), :);

    OldU = [];

    % Assign each entity to cluster
    while true

        AllDist = zeros(N, K);

        for k = 1:K
            AllDist(:, k) = sum((Data - Z(k,:)).^2, 2);
        end

        [~, U] = min(AllDist, [], 2);

        if isequal(OldU, U), break;
    end

    % Update cluster
    for k = 1 : K
        Z(k,:) = mean(Data(U==k, :), 1);
    end

    OldU = U;

end