function cnn = backpropagation(cnn, input_y)
    lrate = 1;
    n = numel(cnn.layers);
    smpsize = size(input_y, 2);
    cnn.error = 1/2 * sum(sum((cnn.output - input_y).^2)) / smpsize;
    
    % calculate delta
    % output layer
    delta{n}.d = (cnn.output - input_y) .* (cnn.output .* (1 - cnn.output));
    % fully-connected layer
    tmp = (cnn.layers{n}.w' * delta{n}.d) .* (cnn.fv .* (1 - cnn.fv));
    for j = 1 : cnn.layers{n-1}.outputmaps
        delta{n-1}.d{j} = reshape(tmp(j,:), 1, 1, smpsize);
    end
    
    for l = (n - 2) : -1 : 1
        % convolution layer
        if strcmp(cnn.layers{l}.type, 'c')
            for j = 1 : cnn.layers{l}.outputmaps
                delta{l}.d{j} = cnn.layers{l}.x{j} .* (1 - cnn.layers{l}.x{j}) .* (expand(delta{l + 1}.d{j}, [cnn.layers{l + 1}.scale cnn.layers{l + 1}.scale 1]) / cnn.layers{l + 1}.scale ^ 2);
            end
        % pooling layer
        elseif strcmp(cnn.layers{l}.type, 'p')
            for i = 1 : cnn.layers{l}.outputmaps
                tmp = zeros(size(cnn.layers{l}.x{1}));
                for j = 1 : cnn.layers{l+1}.outputmaps
                    tmp = tmp + convn(delta{l+1}.d{j}, rot180(cnn.layers{l+1}.k{i}{j}), 'full');
                end
                delta{l}.d{i} = tmp;
            end
        end
    end
    
    % adjust weight
    % output layer
    cnn.layers{n}.w = cnn.layers{n}.w - lrate * (delta{n}.d * (cnn.fv)' / smpsize);
    cnn.layers{n}.b = cnn.layers{n}.b - lrate * mean(delta{n}.d, 2);
    
    for l = 2 : (n - 1)
        if strcmp(cnn.layers{l}.type, 'c') || strcmp(cnn.layers{l}.type, 'f')
            for j = 1 : cnn.layers{l}.outputmaps
                for i = 1 : cnn.layers{l-1}.outputmaps
                    tmp = convn(flipall(cnn.layers{l-1}.x{i}), delta{l}.d{j}, 'valid') / smpsize;
                    cnn.layers{l}.k{i}{j} = cnn.layers{l}.k{i}{j} - lrate * tmp;
                end
                cnn.layers{l}.b{j} = cnn.layers{l}.b{j} - lrate * sum(delta{l}.d{j}(:)) / smpsize;
            end
        end
    end
    
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
    
end