function samples = addsamples(cnn, img, samples)
    w = size(img, 2);
    h = size(img, 1);
    tempgray = img;
    while  w >= 24 && h >= 24 
        tempregion = decarea(tempgray, 24, cnn);
        for i = 1 : size(tempregion, 1)
            tmp = tempgray(tempregion(i,2):tempregion(i,2)+23, tempregion(i,1):tempregion(i,1)+23);
            samples(:,:,end+1) = tmp;
        end
        tempgray = imresize(tempgray, 0.5);
        h = size(tempgray, 1);
        w = size(tempgray, 2);
    end
end