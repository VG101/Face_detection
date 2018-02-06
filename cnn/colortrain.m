clear; close all; clc;
sumCb = 0;
sumCr = 0;
ss = 44;
for i = 1:ss
    RGB = imread(strcat('color/color (', num2str(i), ').jpg'));
    YCbCr = rgb2ycbcr(RGB);
    Cb = YCbCr(:,:,2);
    Cr = YCbCr(:,:,3);
    sumCb = sumCb + mean(Cb(:));
    sumCr = sumCr + mean(Cr(:));
end
fprintf('Cb: %f, Cr: %f\n', sumCb/ss, sumCr/ss);