load test

smpsize = size(test_samples, 3);
cnn = feedforward(cnn, test_samples);
correct = 0;
for i = 1 : smpsize
    if (cnn.output(1, i) > cnn.output(2, i)) &&  (test_labels(1, i) == 1)
        correct = correct + 1;
    end
    if (cnn.output(1, i) < cnn.output(2, i)) &&  (test_labels(1, i) == 0)
        correct = correct + 1;
    end
end

