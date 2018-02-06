clear; 
clc;

load cnn_6
tic;
RGB = imread('faces/image_0124.jpg');
RGB = imresize(RGB,0.3);
gray = rgb2gray(RGB);
%gray = histeq(gray);
[I, Area] = colordetect(RGB);
% wid = size(RGB,2);
% len = size(RGB,1);
% Area = [1 1 wid len];

%Area = expandarea(I, Area);

region = [];
dim = 24;
zoom = 0.75;
for i = 1:size(Area, 1)
    x = Area(i,1);
    y = Area(i,2);
    w = Area(i,3);
    h = Area(i,4);
    %fprintf('%d %d %d %d\n',x,y,w,h);
    tempgray = gray(round(y):round(y+h-1),round(x):round(x+w-1));
    count = 0;
    while ( (w>=dim)&& (h>=dim) )
        tempregion = decarea(tempgray,dim, cnn);
        if size(tempregion, 1) ~= 0
            tempregion(:,:) = floor(tempregion(:,:)*(1/zoom)^(count));
            tempregion(:,1) = tempregion(:,1) + x;
            tempregion(:,2) = tempregion(:,2) + y;
            region = [region;tempregion];
        end
        tempgray = imresize(tempgray,zoom);
        h = size(tempgray, 1);
        w = size(tempgray, 2);
        count = count + 1;
    end
end

% imshow(RGB);
% hold on;
% for i = 1:size(region, 1)
%     rectangle('Position', region(i,:), 'EdgeColor', 'g');
% end
% hold off;

if (size(region,1)>1)
    combine_rect(RGB, region);
elseif  (size(region,1)==1)
    imshow(RGB);
    rectangle('Position',  region(1,:), 'EdgeColor', 'g');
else 
    imshow(RGB);
end
    

toc;