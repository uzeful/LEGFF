function [feat, fImg] = imleguidedfilter(img, r)
% Input:
%           img: input image
%           r: size of local window and radius of structuring element
% Output:
%           feat: feature map
%           fImg: filtered image

    NHSZ = r;
    I = double(img);
    SE = strel('disk', r, 0);
    minImg = imerode(I, SE);
    fImg = imguidedfilter(I, minImg, 'NeighborhoodSize',[NHSZ NHSZ]);
    maxImg = imdilate(fImg, SE);
    fImg = imguidedfilter(fImg, maxImg, 'NeighborhoodSize',[NHSZ NHSZ]);

    feat = double(img) - fImg;
end