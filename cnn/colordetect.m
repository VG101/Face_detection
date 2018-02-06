function [I, Area] = colordetect( RGB )
Size = 20;
Cb_value = 113;
Cr_value = 150;
range = 13;
YCbCr = rgb2ycbcr(RGB);
Cb = YCbCr(:,:,2);
Cr = YCbCr(:,:,3);
h = size(RGB, 1);
w = size(RGB, 2);
I = repmat(RGB, 1);
for r = 1:h
    for c = 1:w
        if Cb(r,c) > (Cb_value+range) || Cb(r,c) < (Cb_value-range) || Cr(r,c) > (Cr_value+range) || Cr(r,c) < (Cr_value-range)
            I(r,c,:) = [0 0 0];
        else
            I(r,c,:) = [255 255 255];
        end
    end
end

%SE = strel('disk',2);
%I = imerode(I, SE);
I = im2bw(I);
stats = regionprops(I, 'BoundingBox');
Area = [];
for i = 1:size(stats, 1)
    bb = stats(i).BoundingBox;
    if (bb(3) >= Size && bb(4) >= Size)
        Area = [Area; bb];
    end
end

end

