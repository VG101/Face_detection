clear; clc;
RGB = imread('myDataBase/one faces/2.jpg');
[I, Area] = colordetect(RGB);

%imshow(I)
imshow(RGB);
hold on;
for i = 1:size(Area, 1)
    rectangle('Position', Area(i,:), 'EdgeColor', 'r');
end
hold off;
