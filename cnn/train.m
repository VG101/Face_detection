clear; clc;
round = 50;
batch_size = 50;

load train_6.mat
smpsize = size(train_samples, 3);
batch_num = smpsize / batch_size;

load cnn_5
%cnn = init_cnn;
for r = 1 : round
    disp(['Round ' num2str(r) '/' num2str(round)]);
    tic;
    kk = randperm(smpsize);
    for l = 1 : batch_num
        batch_x = train_samples(:, :, kk((l - 1) * batch_size + 1 : l * batch_size));
        batch_y = train_labels(:,    kk((l - 1) * batch_size + 1 : l * batch_size));
        
        cnn = feedforward(cnn, batch_x);
        cnn = backpropagation(cnn, batch_y);
    end
    toc;
end
