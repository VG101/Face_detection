function isFace = judge(img, cnn)
    cnn = feedforward(cnn, img);
    if cnn.output(1) > cnn.output(2)
        isFace = 1;
    else
        isFace = 0;
    end
end