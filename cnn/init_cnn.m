function cnn = init_cnn()
    init_mean = 0;
    init_sigma = 0.1;
    cnn.layers = {
        struct('type', 'i', 'outputmaps', 1, 'mapsize', 24)
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)
        struct('type', 'p', 'scale', 2)
        struct('type', 'c', 'outputmaps', 16, 'kernelsize', 3) 
        struct('type', 'p', 'scale', 2)
        struct('type', 'f', 'outputmaps', 32, 'mapsize', 1)
        struct('type', 'o', 'outputmaps', 2)
    };
    n = numel(cnn.layers);

    for l = 2 : (n - 2)
        %convolution layer
        if strcmp(cnn.layers{l}.type, 'c')
            cnn.layers{l}.mapsize = cnn.layers{l-1}.mapsize - cnn.layers{l}.kernelsize + 1;
            for j = 1 : cnn.layers{l}.outputmaps
                for i = 1 : cnn.layers{l-1}.outputmaps
                    cnn.layers{l}.k{i}{j} = normrnd(init_mean, init_sigma, cnn.layers{l}.kernelsize);
                end
                cnn.layers{l}.b{j} = normrnd(init_mean, init_sigma);
            end
        %pooling layer
        elseif strcmp(cnn.layers{l}.type, 'p')
            cnn.layers{l}.mapsize = cnn.layers{l-1}.mapsize / cnn.layers{l}.scale;
            cnn.layers{l}.outputmaps = cnn.layers{l-1}.outputmaps;
        end
    end
    %fully-connected layer
    for j = 1 : cnn.layers{n-1}.outputmaps
        for i = 1 : cnn.layers{n-2}.outputmaps
            cnn.layers{n-1}.k{i}{j} = normrnd(init_mean, init_sigma, cnn.layers{n-2}.mapsize);
        end
        cnn.layers{n-1}.b{j} = normrnd(init_mean, init_sigma);
    end
    %output layer
    cnn.layers{n}.w = normrnd(init_mean, init_sigma, cnn.layers{n}.outputmaps, cnn.layers{n-1}.outputmaps);
    cnn.layers{n}.b = normrnd(init_mean, init_sigma, cnn.layers{n}.outputmaps, 1);
    
end