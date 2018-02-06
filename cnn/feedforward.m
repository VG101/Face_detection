function cnn = feedforward(cnn, input_x)
    n = numel(cnn.layers);  %layers
    cnn.layers{1}.x{1} = input_x;
    smpsize = size(input_x, 3); %sample size
    
    for l = 2 : (n - 1)
        %convolution layer
        if strcmp(cnn.layers{l}.type, 'c') || strcmp(cnn.layers{l}.type, 'f')
            for j = 1 : cnn.layers{l}.outputmaps
                tmp = zeros(cnn.layers{l}.mapsize, cnn.layers{l}.mapsize, smpsize);
                for i = 1 : cnn.layers{l-1}.outputmaps
                    tmp = tmp + convn(cnn.layers{l-1}.x{i}, cnn.layers{l}.k{i}{j}, 'valid');
                end
                cnn.layers{l}.x{j} = sigmf(tmp + cnn.layers{l}.b{j}, [1,0]);
            end
        %pooling layer
        elseif strcmp(cnn.layers{l}.type, 'p') %pooling layer
            scale = cnn.layers{l}.scale;
            for j = 1 : cnn.layers{l}.outputmaps
                tmp = ones(scale) / (scale^2);
                tmp = convn(cnn.layers{l-1}.x{j}, tmp, 'valid');
                cnn.layers{l}.x{j} = tmp(1:scale:end, 1:scale:end, :);
            end
        end
    end
    %output layer
    cnn.fv = zeros(cnn.layers{n-1}.outputmaps, smpsize);
    for i = 1 : cnn.layers{n-1}.outputmaps
        cnn.fv(i, :) = cnn.layers{n-1}.x{i};
    end
    cnn.output = sigmf(cnn.layers{n}.w * cnn.fv + repmat(cnn.layers{n}.b, 1, smpsize), [1,0]);
    
end