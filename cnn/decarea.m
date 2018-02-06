function tempregion = decarea(tempgray,dim, cnn)
step = 2;
tempregion = [];
x = size(tempgray,2);
y = size(tempgray,1);
tests = [];
for i = 1 : step : (y - dim + 1)
    for j = 1 : step : (x - dim + 1)
        temp = tempgray(i:i+dim-1,j:j+dim-1);
        tests(:,:,end+1) = temp;
    end
end
tests(:,:,1) = [];
tcnn = feedforward(cnn, tests);
count = 1;
for i = 1 : step : (y - dim + 1)
    for j = 1 : step : (x - dim + 1)
        if tcnn.output(1, count) > tcnn.output(2, count)
            tempregion = [tempregion; [j i dim dim]];
        end
        count = count + 1;
    end
end