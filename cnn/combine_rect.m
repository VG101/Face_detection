function combined = combine_rect(RGB,Region)
dim = size(Region,1);
combined = [];
for i = 1 : dim
    temp = Region(i,:);
    temp = [temp 1];
    [exist,combined] = faceExist( temp,combined );
    if ( exist==0 )
        combined = [combined;temp];
    end
end

imshow(RGB);
combined = combined(:,1:4);

hold on;
for i = 1:size( combined, 1)
    rectangle('Position',  combined(i,:), 'EdgeColor', 'g');
end
hold off;

end

function [exist,combined] = faceExist( temp,combined )
exist = 0;
left = temp(1);
top = temp(2);
right = left + temp(3);
bottom = top + temp(4);
dim = size(combined,1);
if (dim == 0)
    exist = 0;
    combined = [];
else
    for i = 1 : dim
        left1 = combined(i,1);
        top1 = combined(i,2);
        right1 = left1 + combined(i,3);
        bottom1 = top1 + combined(i,4);
        if (left >= left1 && left <= right1 && top >= top1 && top <= bottom1)
            exist = 1;
            break;
        elseif (right >= left1 && right <= right1 && bottom >= top1 && bottom <= bottom1)
            exist = 1;
            break;
        elseif (left >= left1 && left <= right1 && bottom >= top1 && bottom <= bottom1)
            exist = 1;
            break;
        elseif (right >= left1 && right <= right1 && top >= top1 && top <= bottom1)
            exist = 1;
            break;
        elseif (left1 >= left && left1 <= right && top1 >= top && top1 <= bottom)
            exist = 1;
            break;
        elseif (right1 >= left && right1 <= right && bottom1 >= top && bottom1 <= bottom)
            exist = 1;
            break;
        elseif (left1 >= left && left1 <= right && bottom1 >= top && bottom1 <= bottom)
            exist = 1;
            break;
        elseif (right1 >= left && right1 <= right && top1 >= top && top1 <= bottom)
            exist = 1;
            break;
        end
        if ( exist ==1 )
            weight = temp(5)/( temp(5) + combined(i,5) );
            weight1 = combined(i,5)/( temp(5) + combined(i,5) );
            combined(i,1:4) = weight.*temp(1:4)+weight1.*combined(i,1:4);
            combined(i,5) =  combined(i,5) + 1;
        end  
    end
end









end


